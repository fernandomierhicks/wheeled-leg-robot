# HANDOFF — Fork ODrive 3.6 Web GUI for ODESC V4.2 Compatibility

## Problem

The [MoonLighTingPY/odrive3.6_web_gui](https://github.com/MoonLighTingPY/odrive3.6_web_gui) assumes all boards run ODrive firmware v0.5.6.
Our ODESC V4.2 runs **modified v0.5.1**, which does not have the `usb_cdc_protocol` property (added in v0.5.6).
When the GUI's Apply step tries to write `odrv0.config.usb_cdc_protocol = 3`, the board rejects it and the save fails.

The old v0.5.1 equivalent is `enable_ascii_protocol_on_usb` (a boolean), but we don't even need to set it — USB bulk communication (how the GUI talks to the board) is unaffected by either property.

## Fix — Option A: Skip `usb_cdc_protocol` on Apply

One file change. Removes the property from the wizard so it is never sent to the board.

### Steps

1. **Fork the repo**

   ```bash
   gh repo fork MoonLighTingPY/odrive3.6_web_gui --clone
   cd odrive3.6_web_gui
   git checkout -b fix/odesc-v4.2-usb-cdc
   ```

2. **Edit `frontend/src/utils/odrivePropertyTree.js`**

   Find:
   ```js
   usb_cdc_protocol: {
     name: 'USB CDC Protocol',
     description: 'Protocol for USB virtual COM port',
     writable: true,
     type: 'number',
     valueType: 'Property[ODrive.StreamProtocolType]',
     selectOptions: [
       { value: 0, label: 'Fibre' },
       { value: 1, label: 'ASCII' },
       { value: 2, label: 'Stdout' },
       { value: 3, label: 'ASCII + Stdout' }
     ]
   },
   ```

   Change `writable: true` to `writable: false`:
   ```js
   usb_cdc_protocol: {
     name: 'USB CDC Protocol',
     description: 'Protocol for USB virtual COM port (v0.5.6+ only, skipped on v0.5.1 boards)',
     writable: false,
     type: 'number',
     valueType: 'Property[ODrive.StreamProtocolType]',
     selectOptions: [
       { value: 0, label: 'Fibre' },
       { value: 1, label: 'ASCII' },
       { value: 2, label: 'Stdout' },
       { value: 3, label: 'ASCII + Stdout' }
     ]
   },
   ```

   This prevents the unified registry from generating a write command for this property during Apply. The field still appears in the Inspector as read-only, which is fine.

3. **Build and test**

   ```bash
   cd frontend
   npm install
   npm run dev
   # Open http://localhost:3000, connect ODESC V4.2
   # Run through the 6-step wizard: Power → Motor → Encoder → Control → Interfaces → Apply
   # Verify Apply completes without error
   ```

4. **Commit and push**

   ```bash
   git add frontend/src/utils/odrivePropertyTree.js
   git commit -m "fix: skip usb_cdc_protocol write for v0.5.1 boards (ODESC V4.2)"
   git push -u origin fix/odesc-v4.2-usb-cdc
   ```

## How It Works (architecture reference)

The GUI's config pipeline:

```
InterfaceConfigStep.jsx   →  odriveUnifiedRegistry.js  →  odrivePropertyTree.js
       (UI fields)             (maps config keys to          (master property
                                ODrive write commands)        definitions)
                                       ↓
                              FinalConfigStep.jsx
                              (builds command list,
                               sends via backend to
                               odrivetool over USB)
```

- `odrivePropertyTree.js` defines every ODrive property: name, type, writable, options
- `odriveUnifiedRegistry.js` reads the tree and generates write commands for all `writable: true` properties that have changed
- `FinalConfigStep.jsx` collects those commands, shows a preview, and executes them on Apply
- Setting `writable: false` removes the property from command generation — it never reaches the board

## Other v0.5.1 schema mismatches to watch for

These may also fail on Apply if the GUI tries to write them. Same fix applies — set `writable: false` in the property tree if needed:

| v0.5.6 property (GUI assumes) | v0.5.1 equivalent | Notes |
|---|---|---|
| `usb_cdc_protocol` | `enable_ascii_protocol_on_usb` | Boolean vs int — **this is the one we're fixing** |
| `config.enable_can_a` | `config.enable_i2c_instead_of_can` | Inverted logic |
| `config.enable_brake_resistor` | Not present | May not exist on v0.5.1 |
| `axis.config.can.node_id` | `axis.config.can_node_id` | Flat vs sub-object path |
| `encoder.config.phase_offset` | `encoder.config.offset` | Renamed |
| `motor.is_armed` | `motor.armed_state` | Bool vs int |
| `enable_torque_mode_vel_limit` | `enable_current_mode_vel_limit` | Renamed |

If more properties fail during Apply, check this table and apply the same `writable: false` pattern, or add the v0.5.1 path as an alias in the registry.
