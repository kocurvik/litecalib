// Interactive 3D comparison, one panel per downstream task: monocular depth estimation (mde) and 3D reconstruction (recon).
// Each panel shows two point clouds side by side, the raw fisheye input (left) vs. undistorted with LiteCalib's camera (right); the two canvases share one camera, so orbiting either moves both.
//
// Below them, two collapsed extras: the input photos, and (recon) the reconstruction with the ground-truth camera.
//
// Data: the kocurvik/litecalib_examples repo (its GitHub Pages), web/ written from assets/ by its convert.py:
//   web/manifest.json   {task: {scene: {method: {files: [left, right], gt?, photos: {fisheye, litecalib}, <stats>}}}}
//                       (paths relative to the repo; photos as [thumbnail, original] pairs)
//   web/clouds/<task>/<scene>/<method>_<variant>.bin.gz  gzipped b"LCP2" clouds (format: see parse() below)

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ?examples=<url> loads the data from elsewhere (a local checkout, before it is pushed)
const ROOT = new URLSearchParams(location.search).get("examples") || "https://kocurvik.github.io/litecalib_examples/";
const f1 = (x) => Number(x).toFixed(1);  // one decimal, also for whole numbers (0.0)

// methods / scenes: the buttons, in this order (key = the names in the manifest); a dropdown when they do not fit in one row.
// A method or scene without data in the manifest shows as a disabled button.
// view: the overview the start view glides to, as elevation / azimuth (deg) around the cloud's center and distance in cloud radii.
// The world is the first photo's camera frame (azimuth 0 = behind that camera, y = its up).
const TASKS = {
	mde: {
		// the feed-forward reconstruction models (ff) here get one photo
		methods: [
			{ key: "moge3", name: "MoGe-3" },
			{ key: "unidepth_v2", name: "UniDepthV2" },
			{ key: "unik3d", name: "UniK3D" },
			{ key: "pi3x", name: "Pi3X", ff: true },
			{ key: "mapanything", name: "MapAnything", ff: true },
			{ key: "da3", name: "DA3", ff: true, noK: true },
			{ key: "vggt_omega", name: "VGGT-Omega", ff: true, noK: true },
		],
		scenes: [
			{ key: "berlin", name: "Zip-NeRF Apartment", source: "Zip-NeRF, berlin", note: " The two walls are perpendicular." },
			{ key: "alameda_kitchen", name: "Zip-NeRF Kitchen", source: "Zip-NeRF, alameda" },
			{ key: "berlin_living", name: "Zip-NeRF Living room", source: "Zip-NeRF, berlin" },
			{ key: "nyc_office", name: "Zip-NeRF Office", source: "Zip-NeRF, nyc" },
			{ key: "nyc_kitchen", name: "Zip-NeRF Kitchen 2", source: "Zip-NeRF, nyc" },
			{ key: "london_bedroom", name: "Zip-NeRF Bedroom", source: "Zip-NeRF, london" },
		],
		view: { elev: 80, azim: 0, dist: 2.6 },
		camSize: 0.12,
		labels: ["Raw fisheye", "Undistorted with LiteCalib's camera"],
		stats: () => ["", ""],
		caption: (m, s) => `${m.name} on a single fisheye frame (${s.source}).${s.note || ""}`
			+ (m.ff ? ` ${m.name} is a multi-view model, given this one photo.` : "")
			+ (m.noK ? ` It takes no intrinsics: it gets the undistorted photo alone.` : ""),
	},
	recon: {
		methods: [
			{ key: "pi3x", name: "Pi3X" },
			{ key: "mapanything", name: "MapAnything" },
			// noK: the model takes no intrinsics, so it gets the undistorted photos alone
			{ key: "da3", name: "DA3", noK: true },
			{ key: "vggt_omega", name: "VGGT-Omega", noK: true },
		],
		// input / labels: a scene that is not fisheye (default: "fisheye photos", the task's labels)
		scenes: [
			{ key: "alameda_kitchen", name: "Zip-NeRF Kitchen", source: "Zip-NeRF, alameda", view: { elev: 40, azim: -30, dist: 2.2 } },
			{ key: "berlin_living", name: "Zip-NeRF Living room", source: "Zip-NeRF, berlin", note: " Ceiling removed.", view: { elev: 72, azim: 0, dist: 2.0 } },
			{ key: "berlin", name: "Zip-NeRF Apartment", source: "Zip-NeRF, berlin", view: { elev: 45, azim: 0, dist: 2.6 } },
			{ key: "rotunda_s93", name: "Rotunda", source: "Rotunda", input: "wide-angle photos from phones and GoPros", labels: ["Raw photos", "Undistorted with LiteCalib's cameras"] },
			{ key: "euroc_v2_01_easy_s69", name: "EuRoC", source: "EuRoC, V2_01_easy", input: "wide-angle frames of a drone's camera", labels: ["Raw frames", "Undistorted with LiteCalib's camera"] },
		],
		view: { elev: 40, azim: 0, dist: 2.2 },
		camSize: 0.07,
		labels: ["Raw fisheye", "Undistorted with LiteCalib's camera"],
		stats: (e) => e.maa5 ? e.maa5.map((a) => `pose mAA@5° ${f1(a)}`) : ["", ""],
		caption: (m, s) => `${m.name} on ten ${s.input || "fisheye photos"} (${s.source}).${s.note || ""}`
			+ (m.noK ? ` ${m.name} takes no intrinsics: it gets the undistorted photos alone.` : ""),
		gt: {
			label: "Undistorted with the ground-truth camera",
			stat: (e) => e.maa5_gt != null ? `pose mAA@5° ${f1(e.maa5_gt)}` : "",
			summary: (e) => "Ground-truth camera" + (e.maa5_gt != null ? ` (pose mAA@5° ${f1(e.maa5_gt)})` : ""),
		},
	},
};

const cache = new Map();
let manifestReq = null;

function loadManifest() {
	if (!manifestReq) {
		manifestReq = fetch(ROOT + "web/manifest.json").then((r) => {
			if (!r.ok) throw new Error(`manifest.json: HTTP ${r.status}`);
			return r.json();
		}).catch((err) => { console.error(err); return {}; });
	}
	return manifestReq;
}

// the file's bytes, reporting (received, total) as they come
async function fetchBytes(url, onProgress) {
	const r = await fetch(url);
	if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
	const total = Number(r.headers.get("content-length")) || 0;
	if (!r.body) return new Uint8Array(await r.arrayBuffer());
	const reader = r.body.getReader(), chunks = [];
	let got = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		got += value.length;
		onProgress?.(got, total);
	}
	const out = new Uint8Array(got);
	let o = 0;
	for (const c of chunks) { out.set(c, o); o += c.length; }
	return out;
}

async function gunzip(bytes) {
	if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes.buffer;  // the server decoded it already
	const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
	return new Response(stream).arrayBuffer();
}

function loadCloud(file, onProgress) {
	if (!cache.has(file)) cache.set(file, fetchBytes(ROOT + file, onProgress).then(gunzip).then(parse));
	return cache.get(file);
}

// b"LCP2", little endian: uint32 n_points, n_cams, 0; float32 bbox min[3], max[3]; float32 cams[n_cams][12] (camera-to-world 3x4, row-major); then the points in Morton order, each channel as a plane of wrapping deltas to the
// previous point: xyz int16 (quantized to the bbox) as x, y, z planes, each its low bytes then its high bytes; rgb uint8 as r, g, b planes. Coordinates: three.js (x right, y up, z backward).
function parse(buf) {
	const dv = new DataView(buf), bytes = new Uint8Array(buf);
	if (String.fromCharCode(...bytes.subarray(0, 4)) !== "LCP2") throw new Error("not a LiteCalib point cloud");
	const n = dv.getUint32(4, true), nCams = dv.getUint32(8, true);
	const lo = [0, 1, 2].map((i) => dv.getFloat32(16 + 4 * i, true));
	const hi = [0, 1, 2].map((i) => dv.getFloat32(28 + 4 * i, true));
	let off = 40;
	const cams = [];
	for (let c = 0; c < nCams; c++, off += 48) cams.push(Array.from(new Float32Array(buf.slice(off, off + 48))));
	const pos = new Float32Array(3 * n), rgb = new Uint8Array(3 * n);
	for (let a = 0; a < 3; a++) {
		const low = off + 2 * n * a, high = low + n, scale = (hi[a] - lo[a]) / 65535;
		let v = 0;
		for (let i = 0; i < n; i++) {
			v = (v + (bytes[low + i] | (bytes[high + i] << 8))) & 0xffff;  // the int16 value, mod 2^16
			pos[3 * i + a] = lo[a] + ((v + 32768) & 0xffff) * scale;
		}
	}
	const col = off + 6 * n;
	for (let a = 0; a < 3; a++) {
		let v = 0;
		for (let i = 0; i < n; i++) {
			v = (v + bytes[col + n * a + i]) & 0xff;
			rgb[3 * i + a] = v;
		}
	}
	return { n, pos, rgb, cams };
}

// center and radius of the bulk of the cloud (10th-90th percentile box of a sample), for the start view
function frame(pos) {
	const n = pos.length / 3, step = Math.max(1, Math.floor(n / 20000));
	const axes = [[], [], []];
	for (let i = 0; i < n; i += step) for (let a = 0; a < 3; a++) axes[a].push(pos[3 * i + a]);
	const pct = (v, p) => v[Math.min(v.length - 1, Math.floor(p * v.length))];
	const lo = [], hi = [];
	axes.forEach((v) => { v.sort((x, y) => x - y); lo.push(pct(v, 0.1)); hi.push(pct(v, 0.9)); });
	const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
	const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
	return { center, radius };
}

function cameraLines(cams, size) {
	// a small pyramid per camera: apex at the center, base at depth `size` (the camera looks along -z)
	const w = size * 0.75, h = size * 0.5, d = size;
	const local = [[0, 0, 0], [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d]];
	const edges = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]];
	const verts = [];
	for (const m of cams) {
		const M = new THREE.Matrix4().set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], 0, 0, 0, 1);
		const p = local.map((v) => new THREE.Vector3(...v).applyMatrix4(M));
		for (const [a, b] of edges) verts.push(p[a].x, p[a].y, p[a].z, p[b].x, p[b].y, p[b].z);
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
	return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x9e2b2f }));
}

function h(tag, attrs = {}, children = []) {
	const e = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs)) {
		if (k === "text") e.textContent = v;
		else if (k === "class") e.className = v;
		else e.setAttribute(k, v);
	}
	for (const c of children) e.append(c);
	return e;
}

// a choice: buttons in one row, or a dropdown when they do not fit in it (fit() decides, the CSS shows one of the two)
function choice(label, items, onPick) {
	const group = h("div", { class: "buttons lc-toggle lc-choice", role: "group", "aria-label": label });
	const buttons = items.map((it) => {
		const b = h("button", { class: "button is-small", type: "button", "data-key": it.key, text: it.name });
		b.addEventListener("click", () => onPick(it.key));
		group.append(b);
		return b;
	});
	const options = items.map((it) => h("option", { value: it.key, text: it.name }));
	const select = h("select", { "aria-label": label }, options);
	select.addEventListener("change", () => onPick(select.value));
	const box = h("div", { class: "select is-small lc-select" }, [select]);
	const el = h("div", { class: "lc-control" }, [h("span", { class: "lc-control-label", text: label }), group, box]);
	const fit = () => {
		if (el.hidden) return;
		el.classList.remove("is-compact");
		el.classList.toggle("is-compact", buttons.some((b) => b.offsetTop !== buttons[0].offsetTop));  // wrapped
	};
	return { el, buttons, options, select, fit };
}

// one task panel: method and scene choice, two canvases with a shared camera, a caption
function initTask(mount, task, cfg) {
	const state = { method: null, scene: null, home: null, req: 0, entry: null, f: null };
	let manifest = {};
	const entry = (scene, method) => manifest[scene]?.[method];
	const labelsOf = (scene) => cfg.scenes.find((s) => s.key === scene)?.labels || cfg.labels;

	const methodCtl = choice("Method", cfg.methods, (k) => pick(k, null));
	const sceneCtl = choice("Scene", cfg.scenes, (k) => pick(null, k));
	const reset = h("button", { class: "button is-small is-text lc-reset", type: "button" });
	reset.innerHTML = '<i class="pi pi-refresh"></i>&nbsp;Reset view';
	mount.append(h("div", { class: "lc-controls" }, [h("div", { class: "lc-selects" }, [methodCtl.el, sceneCtl.el]), reset]));

	const makePanel = (label, kind) => {
		const cap = h("figcaption", { class: kind }, [h("strong", { text: label }), h("span", { "data-stat": "" })]);
		return h("figure", { class: "lc-3d-panel is-loading" }, [h("canvas"), cap, h("div", { class: "lc-3d-status", text: "Loading…" })]);
	};
	const panels = cfg.labels.map((label, i) => makePanel(label, i ? "is-ours" : ""));
	const grid = h("div", { class: "lc-3d" }, panels);
	const caption = h("p", { class: "lc-3d-caption" });
	mount.append(grid, caption);

	// the collapsed extras: input photos (filled when opened) and the ground-truth panel (loaded when opened)
	const photosBox = h("details", { class: "lc-details" }, [h("summary", { text: "Input photos" })]);
	const photosBody = h("div", { class: "lc-photos" });
	photosBox.append(photosBody);
	const more = h("div", { class: "lc-more" }, [photosBox]);
	let gtBox = null, gtPanel = null, gtGrid = null;
	if (cfg.gt) {
		gtPanel = makePanel(cfg.gt.label, "is-gt");
		gtGrid = h("div", { class: "lc-3d is-single" }, [gtPanel]);
		gtBox = h("details", { class: "lc-details" }, [h("summary"), gtGrid]);
		more.append(gtBox);
	}
	mount.append(more);

	const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
	const views = [...panels, ...(gtPanel ? [gtPanel] : [])].map((panel) => {
		const canvas = panel.querySelector("canvas");
		let renderer;
		try {
			renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
		} catch (e) {
			panel.querySelector(".lc-3d-status").textContent = "WebGL is not available in this browser.";
			return null;
		}
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.outputColorSpace = THREE.LinearSRGBColorSpace;  // the colors are stored as sRGB already
		renderer.setClearColor(0xfafafa);
		const orbit = new OrbitControls(camera, canvas);
		orbit.addEventListener("change", () => renderAll());
		return { panel, canvas, renderer, orbit, scene: new THREE.Scene() };
	}).filter(Boolean);

	let frameReq = 0;
	function renderAll() {
		cancelAnimationFrame(frameReq);
		frameReq = requestAnimationFrame(() => {
			for (const v of views) {
				const { clientWidth: w, clientHeight: hgt } = v.canvas;
				if (!w || !hgt) continue;
				v.renderer.setSize(w, hgt, false);
				camera.aspect = w / hgt;
				camera.updateProjectionMatrix();
				v.renderer.render(v.scene, camera);
			}
		});
	}

	// the start view: the first photo's own viewpoint (just behind its camera, looking where it looked), then a glide to the overview (cfg.view), as in the video. Any drag or zoom stops the glide.
	const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
	let glide = 0;

	function place(pos, target) {
		camera.position.copy(pos);
		for (const v of views) {
			v.orbit.target.copy(target);
			v.orbit.update();
		}
		renderAll();
	}

	function poses() {
		const { center, radius, elev, azim, dist, cam0 } = state.home;
		const e = THREE.MathUtils.degToRad(elev), a = THREE.MathUtils.degToRad(azim);
		const end = {
			pos: new THREE.Vector3(Math.cos(e) * Math.sin(a), Math.sin(e), Math.cos(e) * Math.cos(a)).multiplyScalar(dist * radius).add(center),
			target: center.clone(),
		};
		const M = new THREE.Matrix4().set(...cam0, 0, 0, 0, 1);
		const start = {
			pos: new THREE.Vector3(0, 0.05 * radius, 0.15 * radius).applyMatrix4(M),
			target: new THREE.Vector3(0, 0, -radius).applyMatrix4(M),
		};
		return { start, end };
	}

	// mode: "glide" (from the photo's viewpoint to the overview), or a still "start" / "end"
	function setView(mode = "glide") {
		if (!state.home) return;
		cancelAnimationFrame(glide);
		camera.near = state.home.radius * 0.01;
		camera.far = state.home.radius * 50;
		const { start, end } = poses();
		if (mode === "start" && !reduceMotion) return place(start.pos, start.target);
		if (mode !== "glide" || reduceMotion) return place(end.pos, end.target);
		const hold = 500, dur = 2200, t0 = performance.now();
		const pos = new THREE.Vector3(), target = new THREE.Vector3();
		const step = (now) => {
			const t = Math.min(1, Math.max(0, (now - t0 - hold) / dur)), k = t * t * (3 - 2 * t);
			place(pos.lerpVectors(start.pos, end.pos, k), target.lerpVectors(start.target, end.target, k));
			if (t < 1) glide = requestAnimationFrame(step);
		};
		glide = requestAnimationFrame(step);
	}
	views.forEach((v) => v.orbit.addEventListener("start", () => cancelAnimationFrame(glide)));

	// run fn once the canvases are well in view (the glide should be seen)
	function whenVisible(fn) {
		const o = new IntersectionObserver((es) => {
			if (es.some((x) => x.isIntersecting)) { o.disconnect(); fn(); }
		}, { threshold: 0.5 });
		o.observe(grid);
	}

	function status(text) {
		panels.forEach((p) => {
			p.querySelector(".lc-3d-status").textContent = text;
			p.classList.toggle("is-loading", !!text);
		});
	}

	// a method / scene is enabled if it has data with some scene / method; picking one whose data is missing with the current choice of the other switches the other to the first one that has it. A choice with a single option is hidden.
	function refreshChoices() {
		for (const [ctl, has, cur] of [
			[methodCtl, (k) => cfg.scenes.some((s) => entry(s.key, k)), state.method],
			[sceneCtl, (k) => cfg.methods.some((m) => entry(k, m.key)), state.scene],
		]) {
			ctl.buttons.forEach((b) => {
				b.disabled = !has(b.dataset.key);
				b.classList.toggle("is-selected", b.dataset.key === cur);
				b.setAttribute("aria-pressed", b.dataset.key === cur);
			});
			ctl.options.forEach((o) => { o.disabled = !has(o.value); });
			ctl.select.value = cur;
			ctl.el.hidden = ctl.options.filter((o) => !o.disabled).length < 2;  // nothing to choose from
			ctl.fit();
		}
	}

	function pick(method, scene) {
		let m = method ?? state.method, s = scene ?? state.scene;
		if (!entry(s, m)) {
			if (method) s = cfg.scenes.find((x) => entry(x.key, m))?.key;
			else m = cfg.methods.find((x) => entry(s, x.key))?.key;
		}
		if (!m || !s || (m === state.method && s === state.scene)) return;
		state.method = m;
		state.scene = s;
		refreshChoices();
		show();
	}

	const mainViews = views.filter((v) => v.panel !== gtPanel), gtView = views.find((v) => v.panel === gtPanel);

	function fill(v, c, f) {
		v.scene.clear();
		const g = new THREE.BufferGeometry();
		g.setAttribute("position", new THREE.BufferAttribute(c.pos, 3));
		g.setAttribute("color", new THREE.BufferAttribute(c.rgb, 3, true));
		v.scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: f.radius * 0.007, vertexColors: true })));
		if (c.cams.length) v.scene.add(cameraLines(c.cams, f.radius * cfg.camSize));
	}

	// the photos of the current scene, raw under the left panel and undistorted under the right one
	let photosFor = null;
	function renderPhotos() {
		const e = state.entry;
		if (!e?.photos || photosFor === e) return;
		photosFor = e;
		photosBody.replaceChildren(...["fisheye", "litecalib"].map((k, i) => h("div", {}, [
			h("span", { class: "lc-control-label", text: labelsOf(state.scene)[i] }),
			h("div", { class: "lc-photo-grid" }, (e.photos[k] || []).map(([thumb, orig]) =>
				h("a", { href: ROOT + orig, target: "_blank", rel: "noopener" }, [h("img", { src: ROOT + thumb, alt: "", loading: "lazy" })]))),
		])));
	}
	photosBox.addEventListener("toggle", () => { if (photosBox.open) renderPhotos(); });

	// the ground-truth cloud, once its box is open and the main clouds are there (for the scale)
	let gtFor = 0;
	async function loadGt() {
		const req = state.req, e = state.entry;
		if (!gtView || !gtBox.open || !e?.gt || !state.f || gtFor === req) return;
		gtFor = req;
		const st = gtPanel.querySelector(".lc-3d-status");
		gtPanel.classList.add("is-loading");
		st.textContent = "Loading…";
		try {
			const c = await loadCloud(e.gt, (got, total) => {
				if (total && req === state.req) st.textContent = `Loading… ${Math.min(99, Math.round((100 * got) / total))} %`;
			});
			if (req !== state.req) return;
			fill(gtView, c, state.f);
			gtPanel.classList.remove("is-loading");
			renderAll();
		} catch (err) {
			if (req === state.req) st.textContent = "Could not load the point cloud.";
			gtFor = 0;
			console.error(err);
		}
	}
	gtBox?.addEventListener("toggle", () => { renderAll(); loadGt(); });

	async function show() {
		const req = ++state.req;
		const e = entry(state.scene, state.method);
		const m = cfg.methods.find((x) => x.key === state.method), s = cfg.scenes.find((x) => x.key === state.scene);
		state.entry = e;
		state.f = null;
		caption.textContent = cfg.caption(m, s, e);
		panels.forEach((p, i) => { p.querySelector("strong").textContent = labelsOf(state.scene)[i]; });
		const stats = cfg.stats(e);
		panels.forEach((p, i) => { p.querySelector("[data-stat]").textContent = stats[i]; });
		photosBox.hidden = !e.photos;
		if (photosBox.open) renderPhotos();
		if (gtBox) {
			gtBox.hidden = !e.gt;
			gtBox.querySelector("summary").textContent = cfg.gt.summary(e);
			gtPanel.querySelector("[data-stat]").textContent = cfg.gt.stat(e);
			gtView?.scene.clear();
			gtPanel.classList.add("is-loading");
		}

		// progress over both files: received / total bytes
		const prog = e.files.map(() => [0, 0]);
		const report = () => {
			if (req !== state.req) return;
			const got = prog.reduce((a, p) => a + p[0], 0), total = prog.reduce((a, p) => a + p[1], 0);
			status(total && prog.every((p) => p[1]) ? `Loading… ${Math.min(99, Math.round((100 * got) / total))} %` : "Loading…");
		};
		status("Loading…");
		let clouds;
		try {
			clouds = await Promise.all(e.files.map((f, i) => loadCloud(f, (got, total) => { prog[i] = [got, total]; report(); })));
		} catch (err) {
			if (req === state.req) status("Could not load the point cloud.");
			console.error(err);
			return;
		}
		if (req !== state.req) return;  // another choice was made meanwhile
		const f = frame(clouds[1].pos);
		const cam0 = clouds[1].cams[0] || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
		state.home = { ...f, ...cfg.view, ...(s.view || {}), cam0 };
		state.f = f;
		mainViews.forEach((v, i) => fill(v, clouds[i], f));
		status("");
		loadGt();
		setView("start");  // the photo's viewpoint until the glide starts
		whenVisible(() => { if (req === state.req) setView(); });
	}

	reset.addEventListener("click", () => setView());
	const resize = new ResizeObserver(renderAll);
	resize.observe(grid);
	if (gtGrid) resize.observe(gtGrid);
	new ResizeObserver(() => { methodCtl.fit(); sceneCtl.fit(); }).observe(mount);

	// load once the panel comes near the screen
	const io = new IntersectionObserver(async (entries) => {
		if (!entries.some((x) => x.isIntersecting)) return;
		io.disconnect();
		manifest = (await loadManifest())[task] || {};
		refreshChoices();
		const s0 = cfg.scenes.find((s) => cfg.methods.some((m) => entry(s.key, m.key)));
		if (!s0) {
			status("Could not load the point clouds.");
			return;
		}
		pick(cfg.methods.find((m) => entry(s0.key, m.key)).key, s0.key);
	}, { rootMargin: "200px" });
	io.observe(mount);
}

export function initClouds() {
	document.querySelectorAll(".lc-task[data-task]").forEach((mount) => {
		const task = mount.dataset.task;
		if (TASKS[task]) initTask(mount, task, TASKS[task]);
	});
}