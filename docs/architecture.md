# Arsitektur CTAMP

## Prinsip batas tanggung jawab

| Layer | Tanggung jawab | Tidak boleh dilakukan |
|---|---|---|
| LLM goal compiler | Mengubah context dan intent menjadi TaskPlan JSON | Menghasilkan joint/trajectory atau mengubah safety |
| Plan layer | Parsing dan empat validation gates | Mengimpor MuJoCo |
| World/scene | Fakta world, scene XML, target slot | Menentukan hasil primitive dari klaim simbolik |
| Task plugin | Semantik task, runtime policy khusus, final goal | Mengakses internal OMPL secara langsung |
| TaskRunner | Dispatch step, hint, recovery, observed verification | Mengetahui semantik internal task/plugin |
| MuJoCo backend | IK, collision, OMPL, trajectory, gripper | Mengubah goal TaskPlan |
| Verifier | Menilai fakta dari live pose | Mempercayai return value primitive sebagai success |

## Alur end-to-end

```text
context markdown
  -> WorldState immutable
  -> LLM call opsional, satu kali
  -> TaskPlan JSON
  -> schema/object/predicate/sequence validation
  -> task plugin validation
  -> slot allocation dan safety validation
  -> resolved RuntimeConfig
  -> generated MuJoCo scene
  -> import backend
  -> TaskRunner
       -> primitive adapter
       -> IK candidate + MuJoCo FK validation
       -> OMPL joint-space planning
       -> execution
       -> observed predicate verification
       -> bounded recovery atau abort
  -> event CSV + summary CSV + run manifest JSON
```

## Import boundary

`backends.mujoco.runtime` mempunyai native initialization side effect.
Karena itu modul ini hanya diimpor oleh CLI setelah context, plan, plugin, slot,
dan runtime config selesai divalidasi. Package `task_planning`, `world`, `plugins`, dan
`execution` dapat di-import untuk unit test tanpa
membuka viewer.

## Konfigurasi

`RuntimeConfig` adalah dataclass immutable yang tersusun dari:

- `ModelConfig`: XML, arm, base, HOME, GRASP_READY, elbow reference, tool axis;
- `IKConfig`: backend, tolerance, candidate dan attempt budget;
- `MotionConfig`: planner, time limit, validity resolution, waypoint/settle;
- `GraspConfig`: cube/cylinder profiles dan approach clearances;
- `SafetyConfig`: obstacle/workspace/contact policies;
- `AdaptiveConfig`: batas aktivasi HintCache;
- `VerificationConfig`: toleransi observed predicates;
- `RecoveryConfig`: batas rebuild stack dan parameter staging;
- `TelemetryConfig`: event path, console, flush, scenario metadata.

TOML overlay diperiksa terhadap schema tersebut. Field asing atau type salah
menghentikan startup. Resolved config lengkap disimpan di run manifest.

## Plugin

Task module di package tepercaya ditemukan berdasarkan suffix `*_task.py`.
Module wajib export `PLUGIN` dengan API `ctamp-task/v2`. Discovery dilakukan
dalam urutan nama module agar reproducible. Package tidak boleh berasal dari
output LLM.
