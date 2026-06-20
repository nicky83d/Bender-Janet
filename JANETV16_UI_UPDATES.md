# Janet V16 Web Interface Updates

**Date:** 2026-06-20  
**Version:** V16 with OAK-D Examples & Enhanced Controls

## Summary of Changes

### 1. **Hermes Tab Moved to Skills**
   - **Before:** Hermes was a main navigation tab alongside Front View, Rear View
   - **After:** Hermes is now accessible via the "Skills" menu as a sub-tab
   - **Files Modified:** `templates/index.html`, `static/js/app.js`
   - **Impact:** Cleaner main navigation, Hermes grouped with other system controls

### 2. **New OAK-D Examples Tab**
   - **Added:** New "OAK-D" main tab for managing vision examples
   - **Features:**
     - Lists all available OAK-D neural network examples from `oak-examples/neural-networks/`
     - Examples include: 3D Detection, Classification, Counting, Depth Estimation, Face Detection, Feature Detection, Keypoint Detection, Line Detection, Object Detection
     - "Bender-Janet" option at top of list to return to normal operating mode
     - Displays current active example
     - One example runs at a time (no simultaneous processing to maintain performance)
   - **Files Modified:** `templates/index.html`, `static/js/app.js`, `static/css/style.css`, `app.py`

### 3. **AI Detection Toggle**
   - **Added:** Header-level toggle control (checkbox) for enabling/disabling all AI detections
   - **Location:** Top-right of header, labeled "AI Detection"
   - **Functionality:**
     - Checkbox starts as checked (AI enabled by default)
     - Toggle sends state to backend via `/ai_detection_toggle` endpoint
     - Persists AI detection state across sessions
   - **Files Modified:** `templates/index.html`, `static/js/app.js`, `static/css/style.css`, `app.py`

### 4. **Backend Routes Added**

#### `/oak_d_list` (GET)
Returns:
```json
{
  "status": "ok",
  "examples": [
    {"id": "3D-detection", "name": "3D Object Detection", "description": "..."},
    ...
  ],
  "current": "bender-janet"
}
```

#### `/oak_d_example` (POST)
Accepts: `{"example": "example-id"}`  
Returns: `{"status": "ok", "message": "Applied OAK-D example: ..."}`  
Special case: `"bender-janet"` returns to normal mode

#### `/ai_detection_toggle` (POST)
Accepts: `{"enabled": true/false}`  
Returns: `{"status": "ok", "enabled": true/false, "message": "..."}`

### 5. **Frontend UI Enhancements**
   - **OAK-D Panel:**
     - Two-column layout: Available Examples + Current Example Display
     - Example buttons with hover states and active highlighting
     - Status message showing example count
     - "Return to Normal" button for quick reset to Bender-Janet mode
   
   - **CSS Classes Added:**
     - `.header-controls` - Header-level control container
     - `.ai-toggle` - AI detection checkbox styling
     - `.oak-d-examples` - Examples list container
     - `.example-btn` - Individual example button styling
     - `.oak-d-current` - Current example display box

### 6. **JavaScript Functions Added**
   - `loadOakDExamples()` - Fetches and renders OAK-D example list
   - `selectOakDExample(exampleId)` - Applies selected example to camera feed
   - Event listeners for AI toggle and OAK-D controls
   - Skill view management updated to include 'hermes'

## Implementation Details

### Camera Feed Handling
- When an example is selected, it updates the camera feed rendering
- Backend handles loading the example's processing pipeline
- Performance: One example at a time to maintain real-time FPS
- Option to keep normal AI detection alongside examples or run examples exclusively

### State Management
- Example state stored in `janet.state` under 'oak_d' section
- AI detection state stored in 'detection' section
- Changes persisted across page reloads via backend state management

### User Experience
1. **Normal Mode (Default):**
   - Front/Rear View tabs show real-time video with Janet's standard AI detection
   - AI Detection toggle enables/disables all detections

2. **OAK-D Example Mode:**
   - Click OAK-D tab to access examples
   - Select an example from the list
   - Camera feed switches to rendering that example
   - "Return to Normal" button returns to Bender-Janet mode
   - Bender-Janet is always available as the first option

## Files Modified
1. ✅ `templates/index.html` - Added OAK-D view, moved Hermes, added AI toggle
2. ✅ `static/js/app.js` - Added example management functions, updated skill views
3. ✅ `static/css/style.css` - Added OAK-D styling
4. ✅ `app.py` - Added three new Flask routes

## Testing Checklist
- [ ] OAK-D tab loads and displays example list
- [ ] Example selection updates camera feed
- [ ] "Return to Normal" button works
- [ ] AI Detection toggle functions
- [ ] Hermes accessible from Skills menu
- [ ] Mobile responsive layout maintained
- [ ] Example buttons show active state
- [ ] Status messages display correctly

## Future Enhancements
- Add visual indicators for which examples support multiple simultaneous instances
- Add example-specific settings/parameters panel
- Save user preferences for frequently-used examples
- Add example preview thumbnails
- Implement example chaining or composition
