# String and Logging Reference

Policy for `QString`, string literals, string-keyed containers, function
signatures, and the diagnostic log system in Blackchirp.

## String Literals

| Form | Type | When to use |
|------|------|-------------|
| `"..."_L1` | `QLatin1StringView` | ASCII-only content. **Default choice** — accepted by any `QAnyStringView` parameter without constructing a `QString`. |
| `u"..."_s` | `QString` | Non-ASCII content (e.g., `u"μs"_s`), or when the call site requires `QString` and the literal is non-ASCII. |
| `"..."_s` | `QString` | Only when the call site genuinely requires a `QString`: widget constructors, `.arg()` receivers, `QStringList` initializers, `QRegularExpression`. |
| `QStringLiteral(...)` | `QString` | **Do not use in new code.** Replace existing occurrences only when the surrounding code is already being edited. |

Do not use `"..."_s` merely because a parameter is `QAnyStringView` — it
constructs a temporary `QString` and defeats the purpose of the view parameter.

`using namespace Qt::Literals::StringLiterals` is pulled in globally via
`data/loghandler.h`, so `_s` and `_L1` are available in all translation units
that include it (directly or transitively).

## Key Declaration Patterns

These patterns replace `static const QString k{"key"};` at namespace scope,
which gives the declaration internal linkage and causes a per-TU copy.

### Pattern A — `inline const QString`

```cpp
inline const QString flow = "Flow%1"_s;
```

Use when the key is used as a `QString` (e.g., `.arg()` is called on it
directly). One heap allocation per process, not per TU.

### Pattern B — `inline constexpr QLatin1StringView`

```cpp
inline constexpr QLatin1StringView trigCh{"trigCh"};
```

Use for ASCII keys consumed by `QAnyStringView` parameters or stored in
`std::map<QString, T, std::less<>>` with heterogeneous lookup. True
`constexpr`; zero runtime cost. Use constructor form `{"..."}` rather than
`"..."_L1` in headers to avoid requiring a `using namespace` declaration.

### Pattern C — `inline constexpr QStringView`

```cpp
inline constexpr QStringView us = u"μs";
```

Same tradeoffs as Pattern B but UTF-16, so non-ASCII keys are safe.

### What not to use

```cpp
// Does NOT compile — QString is not a literal type.
inline constexpr auto k = "key"_s;
```

`QString` is not `constexpr`-eligible in any Qt version this project targets.

## Function Signature Policy

1. **Never pass `QString` by value** unless the callee takes ownership and
   moves. Use `const QString &` or `QAnyStringView` instead.
2. **`QAnyStringView`** for pure lookup, comparison, or passthrough functions.
   Accepts `QString`, `QStringView`, `QLatin1StringView`, and `const char *`
   without a temporary `QString`.
3. **`const QString &`** when the callee needs a `QString` specifically (to
   call `.arg()`, pass to a `const QString &` API, or store as `QString`).

## Container Policy

Use `std::map<QString, T, std::less<>>` for all new `std::map` declarations
keyed on `QString`. The transparent comparator enables heterogeneous lookup
(`find("..."_L1)`, `find(QStringView(...))`) without allocating a temporary
`QString`. Retrofit existing declarations when editing the surrounding code.

`QHash<QString, T>` does not require special declaration — Qt 6 supports
heterogeneous lookup via `qHash(QStringView)` automatically.

## Logging

`LogHandler` is a thread-safe global singleton. Use the free functions:

```cpp
bcLog(u"message"_s);                   // LogHandler::Normal (default)
bcLog(u"message"_s, LogHandler::Debug);
bcDebug(u"detail"_s);
bcWarn(u"warning"_s);
bcError(u"error"_s);
bcHighlight(u"milestone"_s);
```

Inside `HardwareObject` subclasses, prefer the `hw*` helpers that prepend the
device key automatically: `hwLog`, `hwDebug`, `hwWarn`, `hwError`.

Do not use `qDebug()` or `emit logMessage()`. Do not call `LogHandler::instance()`
directly in normal code — use the free functions above.

### Severity guidelines

| Level | Use for |
|-------|---------|
| `Error` | Failures requiring user action or indicating data loss risk |
| `Warning` | Automatically-corrected mismatches the user should know about |
| `Normal` | Connection outcomes, experiment milestones, user-initiated state changes |
| `Highlight` | Major milestones (experiment start/end) |
| `Debug` | Hardware lifecycle, configuration loading, protocol details, parameter traces |

### When to update existing code

Update existing log calls to the correct severity when reviewing a hardware
file for another reason. Do not reclassify messages en masse without reading
them — the difference between a real error and a diagnostic trace requires
per-message judgment.
