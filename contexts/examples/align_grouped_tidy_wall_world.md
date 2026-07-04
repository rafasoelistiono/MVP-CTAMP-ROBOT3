# CONTEXT.MD - Align Grouped Tidy Wall World

## scene
- scene_id: align_grouped_tidy_wall_world
- variant: align_grouped_tidy_wall_world

## table
- x_range: [-0.85, 0.85]
- y_range: [-1.10, 0.90]
- z_top: 0.80
- goal_center: [0.26, 0.34, 0.833]
- goal_area_size_xy: [0.34, 0.47]

## geometry
- cube_size_xyz: [0.066, 0.066, 0.066]

## robot
- id: panda_left
- reach_min_xy: 0.25
- reach_max_xy: 1.50
- base_xy: [0.00, -0.86]
- capabilities: [pick, place]

## objects
- id: a
  class: cube
  color: blue
  pose: [0.24, -0.96, 0.833]
  reachable: true
  near_obstacle: false
- id: b
  class: cube
  color: red
  pose: [0.44, -0.82, 0.833]
  reachable: true
  near_obstacle: false
- id: c
  class: cube
  color: blue
  pose: [0.30, -0.68, 0.833]
  reachable: true
  near_obstacle: false
- id: d
  class: cube
  color: red
  pose: [0.52, -0.54, 0.833]
  reachable: true
  near_obstacle: false
- id: e
  class: cube
  color: blue
  pose: [0.24, -0.40, 0.833]
  reachable: true
  near_obstacle: false
- id: f
  class: cube
  color: red
  pose: [0.44, -0.26, 0.833]
  reachable: true
  near_obstacle: false
- id: g
  class: cube
  color: blue
  pose: [0.34, -0.12, 0.833]
  reachable: true
  near_obstacle: false
- id: h
  class: cube
  color: red
  pose: [0.52, 0.02, 0.833]
  reachable: true
  near_obstacle: false
- id: i
  class: cube
  color: blue
  pose: [0.22, 0.16, 0.833]
  reachable: true
  near_obstacle: false
- id: j
  class: cube
  color: red
  pose: [0.38, 0.30, 0.833]
  reachable: true
  near_obstacle: false
- id: k
  class: cube
  color: blue
  pose: [0.28, 0.44, 0.833]
  reachable: true
  near_obstacle: false
- id: l
  class: cube
  color: red
  pose: [0.20, 0.58, 0.833]
  reachable: true
  near_obstacle: false

## obstacles
- id: frontal_tall_wall
  pose: [0.00, -0.08, 1.60]
  fragile: true
  radius: 0.21
  height: long
  size: [0.08, 0.20, 1.60]

## task
- name: align
- target_objects: [a, b, c, d, e, f, g, h, i, j, k, l]
- description: Move 12 messy red/blue cubes spread from negative to positive y along the right side of a frontal tall wall into two ordered tidy color lanes without using the left side of the wall.

## constraints
- preserve_obstacles: true
- max_retries_per_object: 2
- allowed_predicates: [at, clear, holding, handempty, stable]

## grouped_tidy
- enabled: true
- require_ordered: true
- slot_prefix: tidy_slot
- axis: y
- spacing: 0.075
- row_spacing: 0.110

## tidy_groups
- id: blue_lane
  color: blue
  objects: [a, c, e, g, i, k]
  center: [0.20, 0.34, 0.833]
- id: red_lane
  color: red
  objects: [b, d, f, h, j, l]
  center: [0.32, 0.34, 0.833]

## challenge
- type: frontal_tall_wall
- enabled: true
- obstacle_ids: [frontal_tall_wall]
- require_obstacle_aware_slots: true
- require_motion_probe: true
- inflated_clearance_required: true
- wall_blocks_direct_path: true
- side_corridors_required: true
