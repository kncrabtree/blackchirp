#ifndef PROLOGIXGPIBLAN_H
#define PROLOGIXGPIBLAN_H

#include "gpibcontroller.h"

class PrologixGpibLan : public GpibController
{
    Q_OBJECT
public:
    explicit PrologixGpibLan(QObject *parent = nullptr);

    // GpibController interface
public:
    QString queryTerminator() const;

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();

    // GpibController interface
protected:
    bool readAddress();
    bool setAddress(int a);


};

#endif // PROLOGIXGPIBLAN_H
