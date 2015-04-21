#ifndef VIRTUALINSTRUMENT_H
#define VIRTUALINSTRUMENT_H

#include "communicationprotocol.h"

class VirtualInstrument : public CommunicationProtocol
{
    Q_OBJECT
public:
    explicit VirtualInstrument(QString key, QObject *parent = nullptr);
    ~VirtualInstrument();

    // CommunicationProtocol interface
public:
    bool writeCmd(QString cmd);
    QByteArray queryCmd(QString cmd);

public slots:
    void initialize();
    bool testConnection();
};

#endif // VIRTUALINSTRUMENT_H
