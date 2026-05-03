.. index::
   single: LogHandler
   single: logging; singleton
   single: bcLog
   single: bcWarn
   single: bcError
   single: bcDebug
   single: bcHighlight
   single: MessageCode

LogHandler
==========

``LogHandler`` is the application-wide logging singleton. It accepts messages
from any thread through a thread-safe interface and routes them to the in-app
log display, an on-disk CSV log file, and (when enabled) a separate debug log
file. Most code interacts with it exclusively through the free-function
convenience wrappers; direct calls to the singleton are reserved for connection
setup and experiment-lifecycle management.

Free-function API
-----------------

Five free functions provide thread-safe logging from any context. They are
declared in ``loghandler.h``, which also pulls in
``using namespace Qt::Literals::StringLiterals``, making ``_s`` and ``_L1``
string-literal suffixes available in every translation unit that includes it.

.. code-block:: cpp

   bcLog(u"message"_s);                       // Normal severity (default)
   bcLog(u"message"_s, LogHandler::Warning);  // Explicit severity
   bcDebug(u"detail"_s);                      // Debug severity
   bcWarn(u"condition"_s);                    // Warning severity
   bcError(u"failure"_s);                     // Error severity
   bcHighlight(u"milestone"_s);               // Highlight severity

Prefer these free functions over calling ``LogHandler::instance().log()``
directly. Do not use ``qDebug()`` or ``emit logMessage()`` in new code.

MessageCode severity
--------------------

The ``MessageCode`` enum classifies each log entry:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Use for
   * - ``Normal``
     - Connection outcomes, experiment milestones, user-initiated state changes
   * - ``Warning``
     - Automatically-corrected mismatches the user should know about
   * - ``Error``
     - Failures requiring user action or indicating data-loss risk
   * - ``Highlight``
     - Major milestones such as experiment start and end
   * - ``Debug``
     - Hardware lifecycle events, configuration loading, protocol details,
       parameter traces; written to the debug log file only when debug logging
       is enabled

HardwareObject helpers
----------------------

:cpp:class:`HardwareObject` subclasses use the member helpers ``hwLog``,
``hwWarn``, ``hwError``, and ``hwDebug``, which prepend the device key to every
message before forwarding it to the corresponding free function — see
:cpp:class:`HardwareObject` for their signatures. These helpers are not part of
``LogHandler``'s own API.

Singleton and instance method
------------------------------

``LogHandler::instance()`` returns a reference to the application-wide
singleton. The ``log(text, type)`` instance method is the underlying
implementation that all free functions call. It is also the slot target for
``logMessage()`` and ``logMessageWithTime()`` shim slots that keep legacy
``emit logMessage(...)`` call sites compiling.

On-disk log files
-----------------

``LogHandler`` writes to two log files under the active data path:

- **Main log** — receives all messages of severity ``Normal``, ``Warning``,
  ``Error``, and ``Highlight``.
- **Debug log** — receives ``Debug``-severity messages. Writing to this file
  is enabled only when debug logging is active (controlled at runtime via
  :doc:`applicationconfigmanager`'s ``setDebugLogging()``; the
  ``ApplicationConfigManager::debugLoggingChanged`` signal connects to
  ``LogHandler::setDebugLogging`` at application startup).

``beginExperimentLog(num, msg)`` opens a per-experiment log file for the
duration of an acquisition; ``endExperimentLog()`` closes it. These are called
by ``MainWindow`` and ``BatchManager`` at the experiment lifecycle boundaries.

Display helpers
---------------

``formatForDisplay(text, type, t)`` returns a formatted ``QString`` suitable
for insertion into the in-app log text widget. The ``sendLogMessage`` signal
carries this formatted string; ``iconUpdate`` carries the severity code so the
tab icon can reflect the highest-severity unacknowledged message.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: LogHandler
   :members:
   :undoc-members:

.. doxygenfunction:: bcLog

.. doxygenfunction:: bcDebug

.. doxygenfunction:: bcWarn

.. doxygenfunction:: bcError

.. doxygenfunction:: bcHighlight
