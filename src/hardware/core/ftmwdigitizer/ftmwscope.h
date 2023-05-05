#ifndef FTMWSCOPE_H
#define FTMWSCOPE_H

#include <hardware/core/hardwareobject.h>

#include <QByteArray>

#include <data/experiment/ftmwconfig.h>
#include <hardware/core/ftmwdigitizer/ftmwdigitizerconfig.h>
#include <boost/preprocessor/iteration/iterate.hpp>


namespace BC::Key::FtmwScope {
static const QString ftmwScope{"FtmwDigitizer"};
static const QString bandwidth{"bandwidthMHz"};
}

class FtmwScope : public HardwareObject, protected FtmwDigitizerConfig
{
    Q_OBJECT
public:
    explicit FtmwScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded = true, bool critical = true);
    virtual ~FtmwScope();

signals:
    void shotAcquired(const QByteArray data);

public slots:
    virtual void readWaveform() =0;


    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};

#include BC_STR(BC_FTMWSCOPE_H)

#endif // FTMWSCOPE_H
