#ifndef CUSTOMINSTRUMENT_H
#define CUSTOMINSTRUMENT_H

#include <src/hardware/core/communication/communicationprotocol.h>

class CustomInstrument : public CommunicationProtocol
{
public:
    explicit CustomInstrument(QString key, QString subKey, QObject *parent = nullptr);

public slots:
    void initialize() override;
    bool testConnection() override;
};

#endif // CUSTOMINSTRUMENT_H
