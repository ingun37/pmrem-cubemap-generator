import {
    BoxGeometry,
    CubeCamera,
    CubeReflectionMapping,
    CubeRefractionMapping,
    CubeTexture,
    DataTexture,
    DoubleSide,
    HalfFloatType,
    LinearEncoding,
    LinearFilter,
    LinearMipmapLinearFilter,
    Mesh,
    NoBlending,
    RGBAFormat,
    Scene,
    ShaderMaterial,
    Texture,
    Vector3,
    WebGLCubeRenderTarget,
    WebGLRenderer,
    WebGLRenderTarget,
    WebGLRenderTargetOptions,
} from "three";

// The maximum length of the blur for loop. Smaller sigmas will use fewer
// samples and exit early, but not recompile the shader.
const MAX_SAMPLES = 20;

let _oldTarget: WebGLRenderTarget | null = null;

// Golden Ratio
const PHI = (1 + Math.sqrt(5)) / 2;
const INV_PHI = 1 / PHI;

// Vertices of a dodecahedron (except the opposites, which represent the
// same axis), used as axis directions evenly spread on a sphere.
const _axisDirections = [
    /*@__PURE__*/ new Vector3(1, 1, 1),
    /*@__PURE__*/ new Vector3(-1, 1, 1),
    /*@__PURE__*/ new Vector3(1, 1, -1),
    /*@__PURE__*/ new Vector3(-1, 1, -1),
    /*@__PURE__*/ new Vector3(0, PHI, INV_PHI),
    /*@__PURE__*/ new Vector3(0, PHI, -INV_PHI),
    /*@__PURE__*/ new Vector3(INV_PHI, 0, PHI),
    /*@__PURE__*/ new Vector3(-INV_PHI, 0, PHI),
    /*@__PURE__*/ new Vector3(PHI, INV_PHI, 0),
    /*@__PURE__*/ new Vector3(-PHI, INV_PHI, 0),
];

const _renderTargetParams: WebGLRenderTargetOptions = {
    magFilter: LinearFilter,
    // minFilter: LinearFilter,
    // minFilter: LinearMipmapLinearFilter,
    generateMipmaps: false,
    type: HalfFloatType,
    format: RGBAFormat,
    encoding: LinearEncoding,
    depthBuffer: false,
};
/**
 * This class generates a Prefiltered, Mipmapped Radiance Environment Map
 * (PMREM) from a cubeMap environment texture. This allows different levels of
 * blur to be quickly accessed based on material roughness. It is packed into a
 * special CubeUV format that allows us to perform custom interpolation so that
 * we can support nonlinear formats such as RGBE. Unlike a traditional mipmap
 * chain, it only goes down to the LOD_MIN level (above), and then creates extra
 * even more filtered 'mips' at the same LOD_MIN resolution, associated with
 * higher roughness levels. In this way we maintain resolution to smoothly
 * interpolate diffuse lighting while limiting sampling computation.
 *
 * Paper: Fast, Accurate Image-Based Lighting
 * https://drive.google.com/file/d/15y8r_UpKlU9SvV4ILb0C3qCPecS8pvLz/view
 */

type Direction = "latitudinal" | "longitudinal";

export class PMREMCubeMapGenerator {
    private _renderer: WebGLRenderer;
    private _lodMax: number;
    private _cubeSize: number;
    private _sigmas: number[];
    private _sizeLods: number[];
    private _blurMaterial: ShaderMaterial | null;
    private _pingPongRenderTarget: WebGLCubeRenderTarget | null;

    constructor(renderer: WebGLRenderer) {
        this._renderer = renderer;
        this._pingPongRenderTarget = null;

        this._lodMax = 0;
        this._cubeSize = 0;
        this._sizeLods = [];
        this._sigmas = [];

        this._blurMaterial = null;
    }

    /**
     * Generates a PMREM from an equirectangular texture, which can be either LDR
     * or HDR. The ideal input image size is 1k (1024 x 512),
     * as this matches best with the 256 x 256 cubemap output.
     */
    fromEquirectangular(
        equirectangular: Texture,
        renderTarget: WebGLCubeRenderTarget | null = null
    ) {
        return this._fromTexture(equirectangular, renderTarget);
    }

    /**
     * Disposes of the PMREMGenerator's internal memory. Note that PMREMGenerator is a static class,
     * so you should not need more than one PMREMGenerator object. If you do, calling dispose() on
     * one of them will cause any others to also become unusable.
     */
    dispose() {
        this._dispose();
    }

    // private interface

    _setSize(cubeSize: number) {
        this._lodMax = Math.floor(Math.log2(cubeSize));
        this._cubeSize = Math.pow(2, this._lodMax);
    }

    _dispose() {
        this._blurMaterial?.dispose();

        if (this._pingPongRenderTarget !== null)
            this._pingPongRenderTarget.dispose();
    }

    _cleanup(outputTarget: WebGLRenderTarget) {
        this._renderer.setRenderTarget(_oldTarget);
        outputTarget.scissorTest = false;
    }

    _fromTexture(
        texture: Texture | CubeTexture,
        renderTarget: WebGLCubeRenderTarget | null
    ) {
        if (
            texture.mapping === CubeReflectionMapping ||
            texture.mapping === CubeRefractionMapping
        ) {
            this._setSize(
                texture.image.length === 0
                    ? 16
                    : texture.image[0].width || texture.image[0].image.width
            );
        } else {
            // Equirectangular

            this._setSize(texture.image.width / 4);
        }

        _oldTarget = this._renderer.getRenderTarget();

        const cubeUVRenderTarget = renderTarget || this._allocateTargets();
        this._textureToCubeUV(texture, cubeUVRenderTarget);
        const dataCubeTexture = this._applyPMREM(cubeUVRenderTarget);
        this._cleanup(cubeUVRenderTarget);

        return dataCubeTexture;
    }

    _allocateTargets() {
        const size = this._cubeSize;

        const cubeUVRenderTarget = _createRenderTarget(size, _renderTargetParams);

        if (
            this._pingPongRenderTarget === null ||
            this._pingPongRenderTarget.width !== size
        ) {
            if (this._pingPongRenderTarget !== null) {
                this._dispose();
            }

            this._pingPongRenderTarget = _createRenderTarget(
                size,
                _renderTargetParams
            );

            const { _lodMax } = this;
            // TODO remove log
            ({ sizeLods: this._sizeLods, sigmas: this._sigmas } =
                _createPlanes(_lodMax));

            this._blurMaterial = _getBlurShader();
        }

        return cubeUVRenderTarget;
    }

    _textureToCubeUV(
        texture: Texture,
        cubeUVRenderTarget: WebGLCubeRenderTarget
    ) {
        const renderer = this._renderer;

        const isCubeTexture =
            texture.mapping === CubeReflectionMapping ||
            texture.mapping === CubeRefractionMapping;

        if (isCubeTexture) {
            // TODO just copy cube texture if possible
            throw new Error("from cube texture is not implemented.");
        } else {
            cubeUVRenderTarget.fromEquirectangularTexture(renderer, texture);
        }
    }

    _applyPMREM(cubeUVRenderTarget: WebGLCubeRenderTarget) {
        const renderer = this._renderer;
        const autoClear = renderer.autoClear;
        renderer.autoClear = false;
        const mipmaps: CubeTexture[] = [];
        let previous = cubeUVRenderTarget.texture;
        for (let i = 1; i < this._sigmas.length; i++) {
            const sigma = Math.sqrt(
                this._sigmas[i] * this._sigmas[i] -
                this._sigmas[i - 1] * this._sigmas[i - 1]
            );

            const poleAxis = _axisDirections[(i - 1) % _axisDirections.length];

            // const size = 0;
            const size = this._sizeLods[i];
            const rt = new WebGLCubeRenderTarget(size, _renderTargetParams);
            this._blur(previous, rt, i - 1, i, sigma, poleAxis);
            const dct = makeDataCubeTexture(renderer, size, rt);
            rt.dispose();
            mipmaps.push(dct);
            previous = dct;
        }

        const dataCubeTexture = makeDataCubeTexture(
            renderer,
            this._cubeSize,
            cubeUVRenderTarget
        );
        dataCubeTexture.minFilter = LinearMipmapLinearFilter;
        dataCubeTexture.mipmaps = mipmaps;
        dataCubeTexture.needsUpdate = true;
        renderer.autoClear = autoClear;
        return dataCubeTexture;
    }

    /**
     * This is a two-pass Gaussian blur for a cubemap. Normally this is done
     * vertically and horizontally, but this breaks down on a cube. Here we apply
     * the blur latitudinally (around the poles), and then longitudinally (towards
     * the poles) to approximate the orthogonally-separable blur. It is least
     * accurate at the poles, but still does a decent job.
     */
    _blur(
        previous: CubeTexture,
        current: WebGLCubeRenderTarget,
        lodIn: number,
        lodOut: number,
        sigma: number,
        poleAxis?: Vector3
    ) {
        const pingPongRenderTarget = this._pingPongRenderTarget;

        this._halfBlur(
            previous,
            // TODO: hull exception
            pingPongRenderTarget!,
            lodIn,
            lodOut,
            sigma,
            "latitudinal",
            poleAxis
        );

        this._halfBlur(
            // TODO: hull exception
            pingPongRenderTarget!.texture,
            current,
            lodOut,
            lodOut,
            sigma,
            "longitudinal",
            poleAxis
        );
    }

    _halfBlur(
        targetIn: CubeTexture,
        targetOut: WebGLCubeRenderTarget,
        lodIn: number,
        lodOut: number,
        sigmaRadians: number,
        direction: Direction,
        poleAxis?: Vector3
    ) {
        const renderer = this._renderer;
        // TODO: null exception
        const blurMaterial = this._blurMaterial!;

        if (direction !== "latitudinal" && direction !== "longitudinal") {
            console.error(
                "blur direction must be either latitudinal or longitudinal!"
            );
        }

        // Number of standard deviations at which to cut off the discrete approximation.
        const STANDARD_DEVIATIONS = 3;

        const blurMesh = new Mesh(new BoxGeometry(), blurMaterial);
        const blurUniforms = blurMaterial.uniforms;

        const pixels = this._sizeLods[lodIn] - 1;
        const radiansPerPixel = isFinite(sigmaRadians)
            ? Math.PI / (2 * pixels)
            : (2 * Math.PI) / (2 * MAX_SAMPLES - 1);
        const sigmaPixels = sigmaRadians / radiansPerPixel;
        const samples = isFinite(sigmaRadians)
            ? 1 + Math.floor(STANDARD_DEVIATIONS * sigmaPixels)
            : MAX_SAMPLES;

        if (samples > MAX_SAMPLES) {
            console.warn(
                `sigmaRadians, ${sigmaRadians}, is too large and will clip, as it requested ${samples} samples when the maximum is set to ${MAX_SAMPLES}`
            );
        }

        const weights = [];
        let sum = 0;

        for (let i = 0; i < MAX_SAMPLES; ++i) {
            const x = i / sigmaPixels;
            const weight = Math.exp((-x * x) / 2);
            weights.push(weight);

            if (i === 0) {
                sum += weight;
            } else if (i < samples) {
                sum += 2 * weight;
            }
        }

        for (let i = 0; i < weights.length; i++) {
            weights[i] = weights[i] / sum;
        }

        blurUniforms["envMap"].value = targetIn;
        blurUniforms["samples"].value = samples;
        blurUniforms["weights"].value = weights;
        blurUniforms["latitudinal"].value = direction === "latitudinal";

        if (poleAxis) {
            blurUniforms["poleAxis"].value = poleAxis;
        }

        blurUniforms["dTheta"].value = radiansPerPixel;
        // blurUniforms["mipInt"].value = _lodMax - lodIn;
        blurUniforms["mipInt"].value = lodIn;

        const camera = new CubeCamera(0.1, 1000, targetOut);
        const scene = new Scene();
        scene.add(blurMesh);
        camera.update(renderer, scene);
    }
}

export function _createPlanes(lodMax: number) {
    const sizeLods = [];
    const sigmas = [];

    let lod = lodMax;

    for (let i = 0; i < lodMax; i++) {
        const sizeLod = Math.pow(2, lod);
        sizeLods.push(sizeLod);
        let sigma = 1.0 / sizeLod;
        if (i === 0) {
            sigma = 0;
        }

        sigmas.push(sigma);
        lod--;
    }

    return { sizeLods, sigmas };
}

function _createRenderTarget(size: number, params: WebGLRenderTargetOptions) {
    return new WebGLCubeRenderTarget(size, params);
}

function _getBlurShader() {
    const weights = new Float32Array(MAX_SAMPLES);
    const poleAxis = new Vector3(0, 1, 0);
    return new ShaderMaterial({
        name: "SphericalGaussianBlur",

        defines: {
            n: MAX_SAMPLES,
        },

        uniforms: {
            envMap: { value: null },
            samples: { value: 1 },
            weights: { value: weights },
            latitudinal: { value: false },
            dTheta: { value: 0 },
            mipInt: { value: 0 },
            poleAxis: { value: poleAxis },
        },

        vertexShader: /* glsl */ `
            varying vec3 vOutputDirection;

            void main() {
            	vOutputDirection = position;
				gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4( position, 1.0 );
			}
    `,

        fragmentShader: /* glsl */ `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );
                return textureCubeLodEXT(envMap, normalize(sampleDirection), mipInt).rgb;
				// return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,

        blending: NoBlending,
        depthTest: false,
        depthWrite: false,
        side: DoubleSide,
    });
}


export function makeDataCubeTexture(
    renderer: WebGLRenderer,
    size: number,
    rt: WebGLCubeRenderTarget
) {
    const { format, type, magFilter, minFilter, encoding } = _renderTargetParams;
    const cubeImages = [0, 1, 2, 3, 4, 5].map((face) => {
        const buf = new Uint16Array(size * size * 4);
        renderer.readRenderTargetPixels(rt, 0, 0, size, size, buf, face);
        // return buf;
        // console.log(buf);
        const dt = new DataTexture(
            buf,
            size,
            size,
            format,
            type,
            undefined,
            undefined,
            undefined,
            magFilter,
            undefined,
            undefined,
            encoding
        );
        dt.generateMipmaps = false;
        return dt;
    });
    const ct = new CubeTexture(
        cubeImages,
        undefined,
        undefined,
        undefined,
        magFilter,
        minFilter,
        format,
        type,
        undefined,
        encoding
    );
    ct.needsUpdate = true;

    ct.generateMipmaps = false;
    return ct;
}
