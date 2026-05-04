#ifndef PYTHONHARDWAREBASE_H
#define PYTHONHARDWAREBASE_H

#include <memory>
#include <QString>
#include <QStringList>

#include "pythonprocess.h"

class CommunicationProtocol;

/*!
 * \brief Mixin that turns a HardwareObject subclass into a
 * Python-backed trampoline by handing off lifecycle work to a child
 * Python subprocess.
 *
 * A trampoline is a C++ class that inherits from both a hardware base
 * (e.g. \c AWG, \c Clock, \c FlowController, \c FtmwScope) and
 * PythonHardwareBase via multiple inheritance. The hardware base
 * supplies the Qt slot/signal API; this mixin owns the subprocess and
 * IPC plumbing. Subclasses initialize the mixin with \c d_key and
 * \c d_model, call \c initPythonProcess() from their hardware-base
 * initialize hook, call \c testPythonConnection() from the
 * test-connection hook, delegate \c sleep and \c readSettings to
 * \c pythonSleep() and \c pythonReadSettings(), and translate
 * hardware-specific virtuals into \c pu_process->sendRequest()
 * dispatches with snake_case method names matching the user script.
 * Push-style hardware also calls \c setEnabledProxies() and connects
 * \c pu_process->waveformReceived to a shot handler.
 *
 * Script path, class name, and Python environment are resolved per-
 * profile from HardwareProfileManager. \c startPythonProcess()
 * refuses to start the subprocess if either the script path or the
 * class name is empty rather than substituting a default.
 *
 * \sa PythonProcess, HardwareProfileManager, HardwareObject
 */
class PythonHardwareBase
{
public:
    /*!
     * \brief Construct the mixin with the owning HardwareObject's
     * identity.
     *
     * \param key Hardware key in the form \c "<Type>.<label>"
     *            (e.g., \c "PythonAwg.Default"). Used to look up the
     *            profile's script path, class name, and environment
     *            directory in HardwareProfileManager.
     * \param model Driver class name (e.g., \c "PythonAwg") forwarded
     *              to the Python script as \c self.settings.model.
     */
    explicit PythonHardwareBase(const QString &key, const QString &model);

    /// Stops the subprocess if it is running.
    virtual ~PythonHardwareBase();

    /// Stop the Python subprocess if it is running. Idempotent; safe
    /// to call from the destructor of a subclass.
    void stopProcess();

    /// Returns the human-readable error message produced by the most
    /// recent failed startPythonProcess() or testPythonConnection()
    /// call. Empty when the last operation succeeded.
    QString pythonErrorString() const { return d_pythonErrorString; }

    /*!
     * \brief Resolve the Python interpreter path from a profile's
     * environment directory.
     *
     * Probes \a envPath for the standard venv and conda layouts
     * (\c bin/python3, \c bin/python, \c Scripts/python.exe). Falls
     * back to the literal string \c "python3" — which the operating
     * system resolves through \c PATH — if \a envPath is empty or no
     * interpreter is found inside it.
     *
     * \param envPath Path to a venv or conda environment directory,
     *                or an empty string for the system Python.
     * \return Absolute path to a Python executable, or \c "python3".
     */
    static QString resolvePythonExecutable(const QString &envPath);

    /*!
     * \brief Locate the IPC host script (\c python_hw_host.py) shipped
     * with Blackchirp.
     *
     * Searches the application directory and the standard
     * \c share/blackchirp/ install location.
     *
     * \return Absolute path to the host script, or an empty string
     *         if it is not found in any of the search paths.
     */
    static QString findHostScript();

protected:
    /*!
     * \brief Construct \c pu_process and bind the comm pointer and
     * settings callbacks.
     *
     * Call from the hardware base class's initialize hook. Does not
     * start the subprocess; that happens lazily on the first
     * testPythonConnection() so the registry-driven default-settings
     * pass on profile creation does not spawn a Python interpreter.
     *
     * \param comm Pointer to the trampoline's CommunicationProtocol
     *             (\c p_comm). The pointer is forwarded into the
     *             PythonProcess and refreshed on every
     *             testPythonConnection(); ownership stays with the
     *             HardwareObject.
     * \param getter Callback used to service \c self.settings.get
     *               relay requests. Typically a lambda capturing the
     *               trampoline's SettingsStorage::get.
     * \param setter Callback used to service \c self.settings.set
     *               relay requests. Typically a lambda capturing the
     *               trampoline's SettingsStorage::set (which is
     *               protected, so the lambda is the bridge that lets
     *               the script update persistent settings).
     */
    void initPythonProcess(CommunicationProtocol *comm,
                           PythonProcess::SettingsGetter getter,
                           PythonProcess::SettingsSetter setter);

    /*!
     * \brief Start the subprocess if needed, refresh the comm pointer,
     * and dispatch \c test_connection.
     *
     * Call from the hardware base class's test-connection hook
     * (\c testConnection, \c testClockConnection, \c fcTestConnection,
     * ...). On the first call, lazily starts the subprocess via
     * startPythonProcess(); subsequent calls reuse the running
     * process. Updates the bound comm pointer in case the protocol
     * has been swapped since the last call.
     *
     * \param comm Pointer to the trampoline's CommunicationProtocol
     *             (\c p_comm).
     * \return \c true on success; \c false if the subprocess could
     *         not be started or the script reported a failure
     *         (pythonErrorString() then carries the diagnostic).
     */
    bool testPythonConnection(CommunicationProtocol *comm);

    /*!
     * \brief Launch the Python subprocess for this trampoline.
     *
     * Reads \c pythonScriptPath, \c pythonClassName, and
     * \c pythonEnvPath for the profile keyed by the constructor's
     * \c key argument from HardwareProfileManager. Resolves the
     * interpreter via resolvePythonExecutable() and the host script
     * via findHostScript(), then delegates to PythonProcess::start.
     * Refuses to start (and sets pythonErrorString()) if the script
     * path or class name is empty rather than substituting a
     * placeholder.
     *
     * \return \c true if the process started and completed its
     *         \c _init / \c initialize handshake.
     */
    bool startPythonProcess();

    /*!
     * \brief Forward \c sleep(b) to the Python script.
     *
     * Trampolines that override the HardwareObject sleep() hook
     * delegate to this helper.
     */
    void pythonSleep(bool b);

    /*!
     * \brief Forward \c read_settings to the running Python script.
     *
     * Used by the trampoline's readSettings() override to push
     * post-edit settings back into the script without restarting the
     * subprocess (a restart would re-run \c initialize() and disrupt
     * any connected state).
     */
    void pythonReadSettings();

    /// The owned PythonProcess. Non-null after initPythonProcess()
    /// returns; the subprocess inside it is started lazily on first
    /// testPythonConnection().
    std::unique_ptr<PythonProcess> pu_process;

private:
    const QString d_pyKey;
    const QString d_pyModel;
    QString d_pythonErrorString;
};

#endif // PYTHONHARDWAREBASE_H
