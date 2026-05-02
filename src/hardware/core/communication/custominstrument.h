#ifndef CUSTOMINSTRUMENT_H
#define CUSTOMINSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>



/*!
 * \brief An instrument which has a non-QIODevice communication method
 *
 * Similar to VirtualInstrument, this class leaves the device pointer null.
 * Implementations declare the connection parameters they need from the user
 * (device path, serial number, file handle, etc.) via the
 * \c REGISTER_CUSTOM_COMM macro family in \c hardwareregistration.h. The
 * CommunicationDialog and AddProfileDialog read those descriptors from
 * HardwareRegistry at startup — before any hardware object is constructed —
 * and render the appropriate input widgets. The driver reads the user-supplied
 * values back from the \c BC::Key::Comm::custom group of its SettingsStorage
 * inside \c testConnection().
 */
class CustomInstrument : public CommunicationProtocol
{
public:
    /// Constructs a CustomInstrument with the given identifier.
    /// \param key Identifier passed to the CommunicationProtocol base class.
    /// \param parent QObject parent.
    explicit CustomInstrument(QString key, QObject *parent = nullptr);

public slots:
    /// No-op initialization. CustomInstrument has no QIODevice to create.
    void initialize() override;

    /// Always reports a successful connection. Real verification of a
    /// custom-protocol device happens inside the owning HardwareObject's
    /// own testConnection() override, which inspects the user-supplied
    /// values and attempts whatever vendor-specific handshake is required.
    /// \return Always true.
    bool testConnection() override;

    // CommunicationProtocol interface
public:
    /// Returns nullptr because CustomInstrument has no underlying QIODevice.
    /// \return nullptr.
    QIODevice *_device() override;
};

#endif // CUSTOMINSTRUMENT_H
