# Cara Penggunaan

## 1. Instalasi

Python minimum 3.10. Pada Linux/WSL gunakan `python3` saat membuat environment,
kemudian gunakan `python` setelah environment aktif:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -c "import task_planning, world, execution; print('framework imports OK')"
```

Package framework berada langsung di root repository. Editable install tetap
disarankan agar command dapat dijalankan dari directory lain.

Periksa dependency native:

```bash
python -c "import mujoco; print('mujoco ok')"
python -c "from ompl import base, geometric; print('ompl ok')"
python -c "import pinocchio; print('pinocchio ok')"
```

## 2. Siapkan context

Salin salah satu context pada `contexts/examples/`, lalu isi seluruh section:

- `scene`: scene ID dan variant;
- `table`: bounds, table top, goal center;
- `robot`: ID, base, reach, capabilities;
- `objects`: ID, class, pose, status;
- `obstacles`: pose, ukuran, fragile flag;
- `task`: plugin name, target objects, intent;
- `constraints`: retries dan allowed predicates.

Context parser menghitung ulang reachability dan obstacle proximity. Object ID
yang tidak terdapat pada context akan ditolak dari plan.

## 3. Generate plan

Isi `.env` hanya untuk provider LLM:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=...
```

Kemudian:

```bash
python -m cli.generate_plan --context contexts/examples/ungroup_obs_align_cubes.md --task align --output task_plans/generated
```

Untuk development tanpa network gunakan response file:

```bash
python -m cli.generate_plan \
  --context contexts/examples/ungroup_obs_align_cubes.md \
  --task align \
  --response-file task_plans/examples/ungroup_obs_align_cubes.json \
  --output task_plans/generated
```

Untuk challenge stacked pyramid 6 cube dengan susunan vertikal 3-2-1, gunakan
pasangan context/plan berikut:

```bash
python -m cli.generate_plan \
  --context contexts/examples/ungroup_obs_pyramid_cubes.md \
  --task pyramid \
  --response-file task_plans/examples/ungroup_obs_pyramid_cubes.json \
  --output task_plans/generated
```

Gunakan suffix model pada nama response, misalnya
`ungroup_obs_stack_cubes_qwen_3_coder.json`. `generate_plan` akan otomatis
mempertahankan `qwen_3_coder` pada nama plan generated. `run_simulation` lalu
mengambil label tersebut dari nama plan dan menambahkannya ke semua nama CSV
serta manifest. Untuk input tanpa suffix, label dapat diberikan sekali melalui
`--experiment-label qwen_3_coder`.

## 4. Pilih runtime profile

Built-in profile:

- `conservative`: scene tanpa obstacle;
- `obstacle`: clearance lebih hati-hati dan planning budget lebih besar;
- `auto`: dipilih dari scene variant.

Gunakan TOML untuk eksperimen:

```bash
python -m cli.run_simulation ... --runtime-config configuration/profiles/runtime/verification_strict.toml
```

Jangan menambahkan tuning baru ke `.env`. Tambahkan field typed bila parameter
baru benar-benar diperlukan, lalu override nilainya melalui TOML.

## 5. Jalankan simulasi

```bash
python -m cli.run_simulation \
  --plan task_plans/examples/ungroup_obs_align_cubes.json \
  --context contexts/examples/ungroup_obs_align_cubes.md \
  --scene ungroup_obs \
  --runtime-profile obstacle \
  --viewer
```

Untuk tower, ganti plan dan context dengan pasangan
`ungroup_obs_stack_cubes`. Backend tidak memanggil LLM.

Untuk stacked pyramid, gunakan hasil generated dari command sebelumnya:

```bash
python -m cli.run_simulation \
  --plan task_plans/generated/ungroup_obs_pyramid_cubes_pyramid.json \
  --context contexts/examples/ungroup_obs_pyramid_cubes.md \
  --scene ungroup_obs \
  --runtime-config configuration/profiles/runtime/pyramid.toml \
  --no-viewer
```

Profile `pyramid.toml` menyimpan tuning release dan settle untuk stacked
pyramid. Jangan fine-tune angka release di plugin atau backend; ubah profile
TOML agar run berikutnya tetap reproducible.

Untuk example stack Panda, setiap cube berukuran `0.066 x 0.066 x 0.066 m`
dan `layer_height_m=0.066` adalah nominal symbolic.
Target eksekusi memakai bounding half-extent world-Z aktual dari kedua cube,
sehingga rotasi hasil grasp tidak menyebabkan interpenetrasi pada MuJoCo.

Stack recovery dikendalikan oleh section `[recovery]` pada runtime profile.
Engine memvalidasi prefix tower setelah setiap cube ditempatkan. Cube berikutnya
baru diambil setelah level saat ini valid. Jika placement gagal, cube terkait
diambil ulang, suffix rusak dipindahkan ke staging slot, lalu dibangun kembali
maksimal tiga kali.
Orientation dan velocity tersedia sebagai diagnostic. Hard recovery gate saat
ini memakai XYZ support relation agar kompatibel dengan primitive place yang
belum mengontrol orientasi object secara eksplisit.

## 6. Output

Directory `logs/` berisi:

- `motion_*_events.csv`: detail IK, OMPL, collision, dan trajectory;
- `events_*.csv`: event TaskRunner;
- `task_plan_*.csv`: summary hasil;
- `run_*_manifest.json`: resolved config, hash plan/context, model, plugin, platform.

Directory `models/generated/` berisi scene XML sementara dan diabaikan Git.

Untuk merekam original plan tanpa LLM sebagai reference 100%, jalankan dengan:

```bash
python -m cli.run_simulation ... \
  --plan-source original_no_llm \
  --benchmark-role reference \
  --benchmark-label original-v1
```

Run dari response file memakai label yang sama tetapi
`--plan-source response_file --benchmark-role candidate`. Summary CSV mencatat
`completion_percent`, `plan_source`, `benchmark_role`, `benchmark_label`, dan
`reference_100_percent` agar hasil dapat dibandingkan secara eksplisit.

## 7. Test

```bash
pytest -q
```

Unit/integration tests tidak membuka viewer. Native motion test memerlukan
dependency platform yang lengkap.
