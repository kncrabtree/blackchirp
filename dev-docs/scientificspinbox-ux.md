# ScientificSpinBox UX Improvements

Five targeted improvements to `src/gui/widget/scientificspinbox.{h,cpp}`.

---

## 1. Display Mode Toggle (Auto / Fixed / Scientific)

**Goal:** Prefer fixed-point notation for human-readable magnitudes; let users lock either mode.

Add a `DisplayMode` enum (`Auto`, `Fixed`, `Scientific`) and `d_displayMode` member (default `Auto`).

- **Auto logic in `formatForDisplay`:** if `|value|` is in [10⁻⁶, 10⁶] (and non-zero), format with `'f'`; otherwise format with `'e'` and call `applySuperscript`.
- Expose `displayMode()`/`setDisplayMode(DisplayMode)` as public methods; `setDisplayMode` calls `updateDisplayText` and refreshes the line edit.
- `sizeHint` may need widening slightly to accommodate fixed-point strings like `-123456.000000`.

---

## 2. Context Menu (Mode Selection + Copy Value)

**Goal:** Right-click gives access to display mode and clipboard copy without a separate button.

Override `contextMenuEvent`. Build the standard `QAbstractSpinBox` context menu, then append:

- A **"Display Mode"** submenu with three checkable, mutually exclusive actions: *Auto*, *Fixed*, *Scientific*. The currently active mode is checked. Triggering an action calls `setDisplayMode`.
- A **"Copy Value"** action that calls `QApplication::clipboard()->setText(QString::number(d_value, 'g', 15))`, producing a plain numeric string free of display decoration.

---

## 3. Suffix / Unit Label

**Goal:** Show a unit string (e.g., `" MHz"`, `" ns"`) in the display without interfering with editing or validation.

Add `d_suffix` (`QString`, default empty) with `suffix()`/`setSuffix(const QString &)`.

- `formatForDisplay`: append `d_suffix` to the formatted string.
- `focusInEvent`: strip suffix before switching to edit text (already handled via `updateEditText`, which does not append suffix).
- `valueFromText` and `validate`: strip suffix before parsing/validating.
- `setSuffix` updates `lineEdit()->setMaxLength(MAX_INPUT_LENGTH + d_suffix.length())` and refreshes display.

---

## 4. Invalid Input Visual Feedback

**Goal:** Immediate red-tint feedback when the user's text is not parseable, instead of silent revert on focus-out.

In `onTextChanged`:

- If `d_isEditing`, call `isValidInput(lineEdit()->text())`.
- On invalid: set line edit stylesheet to `background-color: <ThemeColors::getCSSColor(ThemeColors::StatusError, this)>`.
- On valid: clear the stylesheet (empty string restores default).

Clear the stylesheet unconditionally in `focusOutEvent` after reverting the text.

---

## 5. Ctrl-Modifier × 2 Stepping

**Goal:** Ctrl+Up/Down doubles or halves the value (multiplicative), matching scientific intuition better than a fixed percentage.

In `stepBy`, when `Qt::ControlModifier` is held:

- If `d_value` is (near) zero: set to `steps > 0 ? 1.0 : -1.0`.
- Otherwise: `setValue(d_value * std::pow(2.0, steps))`.
- Return immediately (skip the additive path).

Update the class doc-comment to reflect the new Ctrl behaviour.

---

## Implementation Order

1. DisplayMode + `formatForDisplay` changes (needed by context menu)
2. Context menu (depends on DisplayMode; adds Copy)
3. Suffix (independent)
4. Invalid input feedback (independent)
5. Ctrl × 2 stepping (independent)
