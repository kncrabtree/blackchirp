#ifndef LOGHANDLER_H
#define LOGHANDLER_H

#include <atomic>

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QMutex>
#include <QAnyStringView>

using namespace Qt::Literals::StringLiterals;

/*!
 * \brief Application-wide logging singleton.
 *
 * LogHandler routes log messages to the in-app display, an on-disk main log
 * file, and (when enabled) a separate debug log file. All public methods and
 * the free-function wrappers are thread-safe.
 *
 * Normal code should use the free functions (\c bcLog, \c bcDebug, \c bcWarn,
 * \c bcError, \c bcHighlight) rather than calling \c instance() directly.
 * \c HardwareObject subclasses should use the \c hwLog / \c hwWarn /
 * \c hwError / \c hwDebug member helpers instead.
 */
class LogHandler : public QObject
{
    Q_OBJECT
public:
    /*!
     * \brief Severity level for a log message.
     */
    enum MessageCode {
        Normal,    ///<  Connection outcomes, experiment milestones, user-initiated state changes
        Warning,   ///<  Automatically-corrected mismatches the user should know about
        Error,     ///<  Failures requiring user action or indicating data-loss risk
        Highlight, ///<  Major milestones such as experiment start and end
        Debug      ///<  Hardware lifecycle, configuration loading, protocol details; debug-log only
    };
    Q_ENUM(MessageCode)

    /*!
     * \brief Construct a LogHandler.
     * \param logToFile Whether to write messages to disk (true by default).
     * \param parent    Qt parent object.
     */
    explicit LogHandler(bool logToFile = true, QObject *parent = nullptr);
    ~LogHandler();

    /*!
     * \brief Return a reference to the application-wide singleton instance.
     */
    static LogHandler &instance();

    /*!
     * \brief Log a message at the specified severity.
     *
     * Accepts any string type accepted by \c QAnyStringView (QString,
     * QStringView, QLatin1StringView, const char *) without constructing a
     * temporary QString. Thread-safe.
     *
     * \param text Message text.
     * \param type Severity level (default: \c Normal).
     */
    void log(QAnyStringView text, MessageCode type = Normal);

    /*!
     * \brief Format a message for display in the in-app log widget.
     *
     * Prepends a timestamp and a severity indicator to \a text. Called
     * internally before emitting \c sendLogMessage.
     *
     * \param text  Message text.
     * \param type  Severity level.
     * \param t     Timestamp (defaults to the current date/time).
     * \return      Formatted string ready for insertion into a QTextEdit.
     */
    static QString formatForDisplay(const QString &text, MessageCode type,
                                    QDateTime t = QDateTime::currentDateTime());

signals:
    /*!
     * \brief Emitted with the formatted message string for display.
     *
     * Connect to a \c QTextEdit::append slot to update the in-app log view.
     */
    void sendLogMessage(QString);

    /*!
     * \brief Emitted with the severity of each new message.
     *
     * Connect to update a tab icon or status indicator to reflect the
     * highest-severity unacknowledged message.
     */
    void iconUpdate(LogHandler::MessageCode);

public slots:
    /*!
     * \brief Log a message; shim for legacy \c emit logMessage() call sites.
     *
     * Forwards to \c log(). Prefer the free functions in new code.
     *
     * \param text Message text.
     * \param type Severity level (default: \c Normal).
     */
    void logMessage(const QString &text, const MessageCode type = Normal);

    /*!
     * \brief Log a message with an explicit timestamp.
     *
     * Forwards to \c log() using the supplied timestamp instead of the
     * current time. Used when replaying stored messages at a known time.
     *
     * \param text Message text.
     * \param type Severity level (default: \c Normal).
     * \param t    Explicit timestamp.
     */
    void logMessageWithTime(const QString &text, const MessageCode type = Normal,
                            QDateTime t = QDateTime::currentDateTime());

    /*!
     * \brief Open a per-experiment log file for the duration of an acquisition.
     *
     * Called at experiment start. All subsequent messages are also written to
     * the experiment-specific log file until \c endExperimentLog() is called.
     *
     * \param num Experiment number (used to derive the file path).
     * \param msg Initial message written to the experiment log.
     */
    void beginExperimentLog(int num, const QString &msg);

    /*!
     * \brief Close the per-experiment log file opened by \c beginExperimentLog().
     */
    void endExperimentLog();

    /*!
     * \brief Enable or disable writing of Debug-severity messages to the debug log file.
     *
     * Connected to \c ApplicationConfigManager::debugLoggingChanged at
     * application startup so runtime configuration changes propagate
     * automatically.
     *
     * \param enabled True to write Debug messages to the debug log file.
     */
    void setDebugLogging(bool enabled);

private:
    static LogHandler *s_instance;

    std::atomic<int>  d_currentExperimentNum{-1};
    std::atomic<bool> d_logToFile{true};
    std::atomic<bool> d_debugLogging{false};
    QMutex d_fileMutex;

    void doLog(const QString &text, MessageCode type, const QDateTime &t);
    void writeToFile(const QString &text, MessageCode type, const QDateTime &t);
};

/*!
 * \brief Log a message at Normal severity (or an explicit severity).
 *
 * Thread-safe. Forwards to \c LogHandler::instance().log(). This is the
 * preferred logging entry point for most code.
 *
 * \param text Message text (any string type accepted by QAnyStringView).
 * \param type Severity level (default: \c LogHandler::Normal).
 */
void bcLog(QAnyStringView text, LogHandler::MessageCode type = LogHandler::Normal);

/*!
 * \brief Log a message at Warning severity.
 * \param text Message text.
 */
void bcWarn(QAnyStringView text);

/*!
 * \brief Log a message at Error severity.
 * \param text Message text.
 */
void bcError(QAnyStringView text);

/*!
 * \brief Log a message at Debug severity.
 *
 * The message is written to the debug log file only when debug logging is
 * enabled via \c ApplicationConfigManager::setDebugLogging().
 *
 * \param text Message text.
 */
void bcDebug(QAnyStringView text);

/*!
 * \brief Log a message at Highlight severity.
 * \param text Message text.
 */
void bcHighlight(QAnyStringView text);

#endif // LOGHANDLER_H
