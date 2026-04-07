#ifndef PYTHONLIFLASER_H
#define PYTHONLIFLASER_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/core/liflaser/liflaser.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief LifLaser subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonLifLaser launches a Python subprocess (via PythonProcess) that loads a
 * user-written LIF laser driver script. The four private pure virtual methods
 * required by LifLaser — readPos(), setPos(), readFl(), and setFl() — are
 * translated to JSON requests sent over stdin/stdout pipes.
 *
 * The LifLaser base class handles:
 *   - Position validation (min/max range checking) before calling setPos()
 *   - Signal emission (laserPosUpdate, laserFlashlampUpdate)
 *   - beginAcquisition() / endAcquisition() (flashlamp on/off, both final)
 *   - hwPrepareForExperiment() (reads autoDisable setting, final)
 *
 * PythonLifLaser only needs to implement initialize(), testConnection(),
 * and the four IPC-dispatched virtuals.
 */
class PythonLifLaser : public LifLaser, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonLifLaser(const QString &label, QObject *parent = nullptr);

    static QVector<HwConfigParam> configParams();

protected:
    void initialize() override;
    bool testConnection() override;
    void readSettings() override;
    void sleep(bool b) override;
    QStringList forbiddenKeys() const override;

private:
    double readPos() override;
    void setPos(double pos) override;
    bool readFl() override;
    bool setFl(bool en) override;
};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONLIFLASER_H
