#ifndef FTMWSCOPE_H
#define FTMWSCOPE_H

#include "hardwareobject.h"
#include <QVector>
#include <QPointF>
#include <QDataStream>
#include <QTextStream>
#include <QPointF>
#include <QTime>
#include <QStringList>
#include "fid.h"
#include <QTimer>

#if BC_FTMWSCOPE == 1
class Dsa71604c;
typedef FtmwScopeHardware Dsa71604c;
#else
class VirtualFtmwScope;
typedef VirtualFtmwScope FtmwScopeHardware;
#endif

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
    FtmwConfig::ScopeConfig d_configuration;




};

#endif // FTMWSCOPE_H
