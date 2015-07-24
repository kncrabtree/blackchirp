#ifndef FTMWSCOPE_H
#define FTMWSCOPE_H

#include "hardwareobject.h"

#include <QByteArray>

#include "ftmwconfig.h"

class FtmwScope : public HardwareObject
{
    Q_OBJECT
public:
    explicit FtmwScope(QObject *parent = nullptr);

signals:
    void shotAcquired(const QByteArray data);

public slots:
    virtual void readWaveform() =0;


protected:
    BlackChirp::FtmwScopeConfig d_configuration;

};

#ifdef BC_FTMWSCOPE
#if BC_FTMWSCOPE == 1
#include "dsa71604c.h"
class Dsa71604c;
typedef Dsa71604c FtmwScopeHardware;
#elif BC_FTMWSCOPE == 2
#include "mso72004c.h"
class MSO72004C;
typedef MSO72004C FtmwScopeHardware;
#else
#include "virtualftmwscope.h"
class VirtualFtmwScope;
typedef VirtualFtmwScope FtmwScopeHardware;
#endif
#endif

#endif // FTMWSCOPE_H
