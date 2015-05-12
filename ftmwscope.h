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

#if BC_FTMWSCOPE == 1
#include "dsa71604c.h"
class Dsa71604c;
typedef Dsa71604c FtmwScopeHardware;
#else
#include "virtualftmwscope.h"
class VirtualFtmwScope;
typedef VirtualFtmwScope FtmwScopeHardware;
#endif

#endif // FTMWSCOPE_H
