# Riset Performa Codebase CTAMP Robot

Tanggal: 2026-07-12

## Ringkasan Eksekutif

Performa repo ini paling banyak tertahan di jalur runtime MuJoCo/Panda, bukan di struktur web/API. Bukti terkuat ada di artifact run: 12 objek dengan real Panda asset mengambil `2235.789889339001` detik dan hanya menyelesaikan 7/12 objek (`runs/align_grouped_tidy_wall_world_20260709_183909/metrics.json:8-17`). Aksi paling pragmatis: ukur hotspot dengan `cProfile`, kurangi pemanggilan planning/IK yang berulang, batch loop MuJoCo saat aman, dan cegah planner simbolik API/benchmark meledak secara faktorial.

## Sumber Primer

- Repo source dan artifact di path yang dikutip dalam dokumen ini.
- Python `cProfile`/`pstats`: standard library menyarankan `cProfile` untuk profiling program long-running dan bisa diurutkan berdasarkan cumulative time. https://docs.python.org/3/library/profile.html
- Python `functools.cache`/`lru_cache`: cache menyimpan hasil pemanggilan fungsi mahal/I/O-bound dengan argumen hashable dan menyediakan `cache_info()` untuk tuning. https://docs.python.org/3/library/functools.html#functools.lru_cache
- Python `itertools.permutations`/`product`: `permutations` menghasilkan `n! / (n-r)!`, dan `product` setara nested loops/cartesian product. https://docs.python.org/3/library/itertools.html#itertools.permutations
- MuJoCo Python bindings: `mj_step(model, data, nstep=20)` menjalankan beberapa step tanpa reacquire GIL antar-step; bindings juga melepas GIL saat memanggil fungsi C MuJoCo. https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo API: `mj_step` advance simulation; `mj_forward` menghitung pipeline tanpa integrasi waktu; `mj_name2id` mengembalikan id objek bernama. https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-step

## Ranking Aksi

### 1. Tambah profil performa yang bisa diulang sebelum optimasi besar

Impact: tinggi. Effort: kecil.

Repo sudah menulis metrik runtime (`elapsed_time`, `expanded_vertices`, per-object result, IK failures) di `run_scene` (`ctamp/experiments/run_scene.py:542-573`). Artifact run menunjukkan baseline mahal: `elapsed_time` 2235.79 detik, 5 IK failures, 10 retries (`runs/align_grouped_tidy_wall_world_20260709_183909/metrics.json:8-17`, `runs/align_grouped_tidy_wall_world_20260709_183909/metrics.json:307-312`). Gunakan `cProfile` untuk memisahkan waktu di `_next_object`, IK, RRT, dan physics executor. Python docs menyebut `cProfile` cocok untuk long-running programs dan `pstats` bisa sort cumulative time.

Quick win:

- Simpan satu profile run kecil untuk `--max-objects 1` atau `--max-objects 2`.
- Tambah metrik per fase: `xy_plan_time`, `ik_time`, `rrt_time`, `physics_step_time`, `object_selection_time`.

Larger work:

- Buat benchmark fixture dengan scene tetap dan threshold regresi runtime.

Verifikasi:

```bash
python3 -m cProfile -o /tmp/opencode/ctamp_run.prof -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 1 --output /tmp/opencode/ctamp_perf_probe
python3 - <<'PY'
import pstats
pstats.Stats('/tmp/opencode/ctamp_run.prof').strip_dirs().sort_stats('cumulative').print_stats(30)
PY
```

### 2. Cache hasil `plan_xy`/`MotionProbe.probe` selama satu run

Impact: tinggi untuk geometric planning dan object selection. Effort: kecil.

`_next_object()` menghitung `planner.plan_xy(current_xy, start)` dan `planner.plan_xy(start, goal)` untuk setiap objek pending saat sorting (`ctamp/experiments/run_scene.py:219-241`). Setelah objek dipilih, loop utama menghitung transit lagi (`ctamp/experiments/run_scene.py:275-288`). `MotionProbe.probe()` lalu memanggil `path_clear()` untuk direct path dan kandidat corridor (`ctamp/simulation/scene.py:126-156`). Karena input start/goal/clearance/obstacle sama dalam satu run, hasil ini kandidat cache sederhana. Python `functools.lru_cache` memang ditujukan untuk fungsi mahal yang dipanggil berulang dengan argumen hashable, dan punya `cache_info()` untuk mengecek hit/miss.

Quick win:

- Cache di level `MuJoCoMotionPlanner.plan_xy` atau closure di `run_scene`, key dengan `(round(start_x, 4), round(start_y, 4), round(goal_x, 4), round(goal_y, 4))`.
- Laporkan `plan_xy_cache_hits/misses` di `metrics.json`.

Larger work:

- Jika obstacle bergerak, invalidasi cache per perubahan scene state. Saat ini obstacle statis di config dan objek yang dipindah tidak dipakai oleh `MotionProbe.rectangles` (`ctamp/simulation/scene.py:102-112`).

Verifikasi:

```bash
python3 -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 3 --output /tmp/opencode/cache_probe
```

Bandingkan `elapsed_time`, jumlah `plan_xy` calls, dan `cache_info()` sebelum/sesudah.

### 3. Kurangi percobaan IK/RRT yang berulang dan mahal

Impact: sangat tinggi untuk real Panda path. Effort: sedang.

IK solver melakukan loop Damped Least Squares sampai 250 iterasi per target (`ctamp/simulation/panda_ik.py:95-145`). `solve_collision_free()` menambah multistart random sampai 64 default (`ctamp/simulation/panda_ik.py:147-180`). `plan_physical_grasp()` mencoba 5 style grasp, masing-masing bisa membuat kandidat dengan `random_restarts=96`, lalu RRT sampai 4000 iterasi (`ctamp/simulation/panda_ik.py:341-401`). `plan_joint_rrt()` sendiri loop sampai 2500 default dan nearest-node scan list Python (`ctamp/simulation/panda_ik.py:558-631`). Di `run_scene`, selection phase bahkan mencoba IK awal untuk `ranked[:4]` dengan `random_restarts=24` (`ctamp/experiments/run_scene.py:244-257`), lalu main phase mencoba lagi untuk objek yang dipilih (`ctamp/experiments/run_scene.py:317-327`).

Quick win:

- Jangan ulangi grasp probe selection yang sama di main phase; simpan hasil sukses/gagal per `(object_id, object_pose, rounded_start_qpos)`.
- Catat per-object jumlah `solve`, `collision_free_candidates`, dan `plan_joint_rrt` calls di metrics.
- Turunkan default `ranked[:4]` precheck atau aktifkan hanya setelah geometric candidates imbang.

Larger work:

- RRT nearest-node saat ini scan linear setiap extend (`ctamp/simulation/panda_ik.py:581-583`). Jika profile membuktikan bottleneck, gunakan struktur nearest-neighbor dari dependency yang sudah ada (`scikit-learn` terpasang di `pyproject.toml:20`) atau batch NumPy distance. Jangan tambah dependency baru.
- Buat cache kegagalan IK dengan TTL per scene revision agar object yang sudah gagal dengan pose sama tidak dicoba ulang dengan budget besar.

Verifikasi:

```bash
python3 -m cProfile -o /tmp/opencode/ik.prof -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 1 --output /tmp/opencode/ik_probe
```

Metrik target: jumlah RRT call turun, total `elapsed_time` turun, success/placement tidak regresi.

### 4. Batch MuJoCo stepping saat tidak butuh observasi per-step

Impact: tinggi untuk physics execution. Effort: kecil-sedang.

`MuJoCoBackend.step(n)` saat ini loop Python dan memanggil `mj_step` satu per satu (`ctamp/simulation/mujoco_backend.py:39-42`). `PandaPhysicsExecutor.settle()`, `_ramp_gripper()`, `command_arm()`, dan `follow_joint_path()` juga banyak loop step Python (`ctamp/simulation/panda_physics.py:53-61`, `ctamp/simulation/panda_physics.py:75-97`, `ctamp/simulation/panda_physics.py:99-111`). MuJoCo Python docs menyebut top-level `mj_step` punya argumen `nstep`; `mj_step(model, data, nstep=20)` menjalankan beberapa physics step tanpa acquire GIL di antara step, berbeda dari loop Python.

Quick win:

- Ubah `MuJoCoBackend.step(n)` jadi panggil `mj_step(self.model, self.data, nstep=n)` saat viewer tidak aktif.
- Untuk `settle(steps)` tanpa viewer, batch langsung.
- Untuk `_ramp_gripper`, tetap per-step jika target ctrl berubah tiap step; jangan optimasi sebelum profile.

Larger work:

- Pisahkan mode `interactive_viewer` vs headless. Di viewer, `sync()` dan sleep tetap perlu per-step (`ctamp/simulation/panda_physics.py:53-57`). Di headless, batch lebih aman.

Verifikasi:

```bash
python3 -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 1 --output /tmp/opencode/nstep_probe
```

Bandingkan `elapsed_time`, final pose, dan `physical_tidy_success`.

### 5. Ganti planner simbolik API/benchmark yang enumerate semua permutasi

Impact: sangat tinggi untuk API/benchmark dengan object count naik. Effort: sedang-besar.

`SymbolicTaskPlanner.solve()` membangun branch untuk setiap `itertools.permutations(obj_ids)` dan setiap `itertools.product(available_arms, repeat=len(perm))` (`ctamp/planning/symbolic.py:46-59`). Docstring-nya juga menyatakan semua ordering dan arm combos dienumerasi (`ctamp/planning/symbolic.py:30-35`). Python docs mendefinisikan panjang full permutations sebagai `n!`; `product` setara nested loop cartesian product. Dengan dua arm, search space kira-kira `n! * 2^n`. API `/planning/run` memakai path ini (`ctamp/api/planning_manager.py:24-42`), dan benchmark menjalankan baseline/offline/online untuk tiap episode (`ctamp/benchmark/episode_runner.py:79-107`).

Quick win:

- Tambah guard object count untuk API path atau pakai ordered branch seperti CLI `_build_ordered_tmm()` (`ctamp/experiments/run_scene.py:33-75`) saat request object count melewati threshold kecil.
- Di benchmark, jalankan object count kecil dulu; default CLI `1,2,3,4,5` masih aman-ish, tetapi growth eksponensial/faktorial harus eksplisit (`ctamp/benchmark/cli.py:15-18`).

Larger work:

- Implement beam search/branch-and-bound yang menghasilkan sebagian order, bukan semua order upfront.
- Perbaiki learned benchmark: `_run_learned()` membuat `HeuristicPathEstimator` tetapi tetap memanggil `BaselinePlanner` tanpa heuristic estimator (`ctamp/benchmark/episode_runner.py:135-162`), jadi angka offline/online tidak mengukur planner learned sebenarnya.

Verifikasi:

```bash
python3 -m ctamp.benchmark.cli -n 1 -o 1,2,3 --no-plots -d /tmp/opencode/ctamp_bench
```

Metrik target: nodes/edges generated, wall time per object count, memory peak.

### 6. Hindari compile/load MuJoCo model berulang di fallback path

Impact: sedang-tinggi jika fallback sering terpanggil. Effort: sedang.

`run_scene` build XML sekali dan load live backend (`ctamp/experiments/run_scene.py:119-124`). Untuk real Panda, planning backend kedua juga load model dari XML string (`ctamp/experiments/run_scene.py:178-180`). Di `_move_arm_to_safe_pose()`, saat direct return home gagal, code membuat `MuJoCoBackend()` baru dan `load_model(xml_string=xml)` lagi (`ctamp/experiments/run_scene.py:191-207`). MuJoCo docs menyebut `MjModel.from_xml_string` membuat model dari XML string; proses compile model ini layak diukur sebelum dipanggil berulang.

Quick win:

- Profil dulu jumlah masuk fallback `_move_arm_to_safe_pose()`.
- Jika fallback sering, reuse planning backend/model atau copy/reset data daripada compile ulang XML setiap fallback.

Larger work:

- Pisahkan immutable compiled model dari mutable `MjData`; MuJoCo Python docs menunjukkan `MjData(model)` adalah data runtime untuk model yang sama.

Verifikasi:

```bash
python3 -m cProfile -o /tmp/opencode/load_model.prof -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 2 --output /tmp/opencode/load_model_probe
```

Cari cumulative time di `load_model`, `from_xml_string`, dan `_move_arm_to_safe_pose`.

### 7. Cache MuJoCo name lookups jika profile menunjukkan overhead

Impact: rendah-sedang. Effort: kecil.

`MuJoCoBackend.get_body_pose()` dan `set_body_pose()` memanggil `_body_id()` setiap call (`ctamp/simulation/mujoco_backend.py:44-63`), dan `_body_id()` memanggil `mj_name2id` (`ctamp/simulation/mujoco_backend.py:86-90`). `run_scene` memanggil get/set body pose dalam sync planning scene untuk semua object (`ctamp/experiments/run_scene.py:183-190`) dan saat object berhasil (`ctamp/experiments/run_scene.py:492-506`). MuJoCo API docs menyatakan `mj_name2id` mengambil id object bernama; cache id per backend adalah perubahan kecil, tetapi jangan prioritas sebelum IK/physics terukur.

Quick win:

- Cache `body_name -> id`, `equality_name -> id`, dan joint qpos address di backend/executor.

Larger work:

- Pakai MuJoCo named access jika lebih jelas, tetapi ukur karena docs hanya menjamin named access O(1), bukan gratis.

Verifikasi:

```bash
python3 -m cProfile -o /tmp/opencode/name_lookup.prof -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 1 --output /tmp/opencode/name_lookup_probe
```

## Quick Wins

- Jalankan `cProfile` untuk `--max-objects 1` dan urutkan cumulative time.
- Cache `plan_xy` dalam satu run dan expose `cache_info()` di metrics.
- Batch `MuJoCoBackend.step(n)` memakai `mj_step(..., nstep=n)` untuk headless/non-viewer.
- Guard API symbolic planner saat object count melewati threshold kecil.
- Tambah per-phase timing di `metrics.json`.

## Pekerjaan Lebih Besar

- Cache/short-circuit IK/RRT result per pose/start state.
- Replace exhaustive symbolic graph generation untuk API/benchmark dengan beam/branch-and-bound.
- Reuse compiled MuJoCo model/data di fallback planning backend.
- Buat benchmark regresi performa dengan scene tetap, object count bertahap, dan artifact CSV/JSON.

## Perintah Verifikasi

Sudah dijalankan:

```bash
python -m pytest
# gagal: /bin/bash: line 1: python: command not found

python3 -m pytest
# 2 passed in 2.27s
```

Belum dijalankan karena full real-Panda run artifact yang ada sudah 2235.79 detik dan scope riset ini tidak mengubah app code:

```bash
python3 -m cProfile -o /tmp/opencode/ctamp_run.prof -m cli.run_simulation --config configs/scenes/align_grouped_tidy_wall_world.yaml --max-objects 1 --output /tmp/opencode/ctamp_perf_probe
python3 -m ctamp.benchmark.cli -n 1 -o 1,2,3 --no-plots -d /tmp/opencode/ctamp_bench
```
