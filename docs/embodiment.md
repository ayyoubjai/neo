# EMBODIMENT Design Note

## Why this fits the system

The current system already has two different kinds of "modes":

- Interface modes: `text`, `audio`, `vision`, `local`, `telegram`
- Orchestrator requested modes: `GENERAL`, `AGENT`, `COMPLEX`, `COGNITION`, `ART`, `ASI`

These two families do different jobs.

- Interface modes decide how the system receives and emits interaction.
- Orchestrator modes decide how the system thinks and executes.

Because of that, `EMBODIMENT` should not start as another peer to `GENERAL` or `COGNITION`.
It fits better as a subsystem below those modes:

- It discovers hardware and peripherals.
- It models what the current host can sense and control.
- It exposes those capabilities as tools and runtime adapters.
- Existing reasoning modes then decide when to use those tools.

In short:

- `EMBODIMENT` is a body/capability layer.
- `AGENT`, `COMPLEX`, and `COGNITION` remain reasoning/execution layers.

## Current system shape

The repository already contains the first pieces of embodiment:

- `src/voice_daemon/`: local audio and vision senses
- `src/runtime_core/`: local runtime composition for senses
- `src/tool_runtime/tools.py`: UI control tools such as screenshot, click, type, drag, scroll
- `config/tool_registry.json`: tool metadata, permissions, schemas
- `src/orchestrator/main.py`: requested-mode dispatch and tool execution loops

That means the system is not missing embodiment entirely.
It is missing a unified layer that says:

- What hardware is present?
- What can this host do right now?
- Which tools should exist for this host?
- What permissions and risks apply to each device action?
- How do we support phones, desktops, tablets, and robots through one model?

## Recommended definition

`EMBODIMENT` should own four responsibilities.

### 1. Device discovery

Detect and describe available entities such as:

- display
- camera
- microphone
- speaker
- keyboard
- mouse / touch input
- filesystem access
- network access
- location / GPS
- battery / power state
- bluetooth
- USB / serial devices
- robot controllers
- sensors and actuators

The output should be structured, for example:

- device id
- device kind
- host platform
- transport
- capabilities
- health / availability
- safety level
- required permissions

### 2. Capability normalization

Do not model raw hardware first. Model stable capabilities first.

Examples:

- `screen.capture`
- `screen.inspect`
- `pointer.move`
- `pointer.click`
- `keyboard.type`
- `audio.listen`
- `audio.speak`
- `camera.capture`
- `camera.stream`
- `file.read_external`
- `robot.move_joint`
- `robot.gripper.open`

Then map platform-specific adapters into those stable capabilities.

This is the key design choice that lets one system work across:

- laptop
- phone
- tablet
- kiosk
- Raspberry Pi
- robot host

### 3. Tool exposure

For each capability, expose tools with:

- strict schema
- permission tier
- safety policy
- timeout/resource limits
- platform availability checks

Some tools should be always present but return `unavailable` when unsupported.
Others can be registered dynamically only when the capability exists.

### 4. Embodiment state

Keep a shared state snapshot for the active host:

- available devices
- active permissions
- current foreground window/app
- screen geometry
- battery/network state
- attached robot state
- recently failed capability calls

This state should be queryable by the orchestrator so planning can adapt to the actual body.

## Recommended architecture

Add a new package:

- `src/embodiment/`

Suggested internal structure:

- `src/embodiment/manager.py`
  - lifecycle entry point
- `src/embodiment/models.py`
  - device and capability schemas
- `src/embodiment/registry.py`
  - discovered devices and active capabilities
- `src/embodiment/discovery/`
  - platform/device detectors
- `src/embodiment/adapters/`
  - platform-specific implementations
- `src/embodiment/tools.py`
  - embodiment-backed tool handlers
- `src/embodiment/policy.py`
  - safety classes and permission rules

Core model:

- `Device`: what exists physically or logically
- `Capability`: what can be done
- `Adapter`: how a capability is implemented on this platform
- `ActionSpec`: tool-safe callable action derived from a capability

## Important design rule

Do not make tools directly equal to devices.

That creates a bad API because devices vary too much.

Prefer:

- device -> capabilities -> tools

Example:

- Device: `Logitech Brio`
- Capability: `camera.capture`
- Tool: `camera.capture_frame`

Example:

- Device: `Android touchscreen`
- Capability: `pointer.tap`, `pointer.swipe`, `screen.capture`
- Tools: `ui.tap`, `ui.swipe`, `ui.screenshot`

Example:

- Device: `robot_arm_1`
- Capability: `robot.pose.move`, `robot.gripper.open`
- Tools: `robot.move_pose`, `robot.set_gripper`

## Should EMBODIMENT be a mode?

Not initially.

If you add it as a new requested mode now, you must update:

- session requested-mode validation
- orchestrator mode normalization
- router model contract
- router examples
- dispatch logic
- mode logs
- tests for routing and mode persistence

That is a lot of surface area for something whose real job is hardware abstraction.

Better path:

1. Build `EMBODIMENT` as a subsystem.
2. Let `AGENT` / `COMPLEX` / `COGNITION` use embodiment tools.
3. Only later consider an `EMBODIMENT` mode if you discover a distinct reasoning pattern such as:
   - "inspect host hardware"
   - "reconfigure devices"
   - "calibrate robot body"
   - "diagnose embodiment failures"

If such a mode appears, it should be a diagnostic/planning mode for the body, not the body itself.

## Cross-platform strategy

Use adapters per environment:

- `desktop_windows`
- `desktop_linux`
- `desktop_macos`
- `android`
- `ios`
- `embedded_linux`
- `robot_ros`

Each adapter should declare:

- supported capabilities
- implementation status
- runtime requirements
- permission prompts
- safety class

## Explicit First, LLM Fallback Second

The low-level embodiment path should default to explicit adapters and tool handlers.

Examples:

- desktop UI via Python UI libraries
- Android phone control via ADB or local platform APIs
- robot control via ROS or vendor SDK bridges

This keeps the primary embodiment path:

- deterministic
- testable
- permission-aware
- auditable

When the system detects a device body but does not have an available explicit adapter, the fallback strategy is:

1. expose the missing capability in the registry as `configured`
2. expose fallback candidates through `embodiment.list_fallbacks`
3. let the orchestrator use its existing `create_tool` loop to generate a bounded low-level tool implementation
4. restrict that fallback to local host/device APIs only, with no network access

So the runtime rule becomes:

- explicit embodiment adapters are the default path
- LLM-generated embodiment tools are the fallback path

This avoids hard-coding desktop assumptions into the whole system.

## Safety and permissions

Embodiment introduces much higher risk than file or text tools.

Separate capability classes at least into:

- observe
  - screenshot, camera frame, microphone input, sensor read
- interact
  - click, type, swipe, focus window
- actuate
  - robot motion, power control, external hardware commands

Recommended rule:

- observe = medium by default
- interact = high by default
- actuate = very high by default

For robots, add:

- dry-run mode
- workspace bounds
- velocity/force caps
- emergency stop support
- explicit arm/disarm state

## Implementation phases

### Phase 1: Inventory and model

- Create embodiment data models.
- Add discovery for local desktop hardware already partly used by the repo.
- Expose a read-only tool like `embodiment.describe_host`.

### Phase 2: Unify existing device code

- Wrap current voice, vision, and UI features behind embodiment capabilities.
- Keep existing tool IDs working.
- Internally route them through embodiment adapters.

### Phase 3: Dynamic tool exposure

- Generate available tool surface from detected capabilities.
- Mark unsupported actions clearly.
- Surface capability context to the orchestrator before planning.

### Phase 4: External and robotic devices

- Add serial / bluetooth / network-connected hardware adapters.
- Add robot-safe action classes and policy gates.

## First concrete features I would build

In this repo, the highest-value starting point is:

1. `embodiment.describe_host`
   - enumerate displays, audio devices, camera availability, UI control support
2. `embodiment.list_capabilities`
   - return normalized capabilities and safety tiers
3. `embodiment.get_state`
   - foreground app, screen size, battery/network if available
4. Refactor existing `ui.*`, audio, and vision pieces to register through embodiment

That gives you a coherent base without committing to robot APIs too early.

## Verdict

The idea is strong.

The main correction is architectural:

- do not define `EMBODIMENT` first as a reasoning mode
- define it first as the system's hardware and capability layer

If you do that, the system can scale from:

- local PC
- phone
- multimodal edge device
- robot-connected host

without rewriting the orchestrator every time a new body appears.
