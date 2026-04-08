#ifndef PYTHONHARDWAREBASE_H
#define PYTHONHARDWAREBASE_H

#include <memory>
#include <QString>
#include <QStringList>

#include "pythonprocess.h"

class CommunicationProtocol;

/*!
 * \brief Mixin base class providing common Python hardware subprocess management
 *
 * PythonHardwareBase extracts the boilerplate shared across all PythonXxx
 * classes: launching the Python subprocess, finding the host script,
 * sleep/readSettings dispatch, process initialization, and test connection
 * logic.
 *
 * Concrete Python hardware classes inherit from both their hardware base
 * class (e.g., AWG, Clock) and PythonHardwareBase via multiple inheritance.
 * They must call initPythonProcess() from their initialize override and
 * testPythonConnection() from their testConnection override.
 *
 * The constructor takes the HardwareObject's key and model strings, which
 * are used for profile manager lookups and process identification.
 */
class PythonHardwareBase
{
public:
    explicit PythonHardwareBase(const QString &key, const QString &model);
    virtual ~PythonHardwareBase();

    /*!
     * \brief Returns forbidden keys contributed by the Python mixin
     *
     * Concrete classes should append these to their own forbiddenKeys() list.
     * Currently returns commType and model keys, since these are managed
     * by the runtime hardware config for Python hardware.
     */
    static QStringList pythonForbiddenKeys();

    /*!
     * \brief Resolve the Python executable from an environment directory
     *
     * Checks for venv/conda layout (bin/python3, bin/python, Scripts/python.exe).
     * Falls back to "python3" if envPath is empty or no interpreter is found.
     *
     * \param envPath Path to venv or conda environment directory (may be empty)
     * \return Resolved executable path or "python3"
     */
    static QString resolvePythonExecutable(const QString &envPath);

protected:
    /*!
     * \brief Initialize the PythonProcess and wire up callbacks/signals
     *
     * Call this from the concrete class's initialize()/initializeClock()/etc.
     *
     * \param comm Pointer to the CommunicationProtocol (p_comm)
     * \param getter Settings getter callback (wrapping SettingsStorage::get)
     * \param setter Settings setter callback (wrapping SettingsStorage::set)
     */
    void initPythonProcess(CommunicationProtocol *comm,
                           PythonProcess::SettingsGetter getter,
                           PythonProcess::SettingsSetter setter);

    /*!
     * \brief Test connection to the Python subprocess
     *
     * Starts the process if not running, updates comm, sends test_connection.
     * Call this from the concrete class's testConnection()/testClockConnection()/etc.
     *
     * \param comm Pointer to the CommunicationProtocol (p_comm)
     * \return true if connection test succeeded; on failure, errorString() has details
     */
    bool testPythonConnection(CommunicationProtocol *comm);

    /*!
     * \brief Start the Python subprocess
     *
     * Looks up script path and class name from HardwareProfileManager.
     * Returns false and sets errorString if either is empty.
     *
     * \return true if process started successfully
     */
    bool startPythonProcess();

    /*!
     * \brief Find the python_hw_host.py script
     * \return Path to host script, or empty string if not found
     */
    static QString findHostScript();

    /*!
     * \brief Dispatch sleep to the Python subprocess
     * \param b true to enter sleep mode, false to wake
     */
    void pythonSleep(bool b);

    /*!
     * \brief Dispatch read_settings to the Python subprocess
     */
    void pythonReadSettings();

    /*!
     * \brief Returns the error string from the last failed operation
     */
    QString pythonErrorString() const { return d_pythonErrorString; }

    std::unique_ptr<PythonProcess> pu_process;

private:
    const QString d_pyKey;
    const QString d_pyModel;
    QString d_pythonErrorString;
};

#endif // PYTHONHARDWAREBASE_H
