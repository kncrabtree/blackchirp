#ifndef CUSTOMINSTRUMENT_H
#define CUSTOMINSTRUMENT_H

#include <src/hardware/core/communication/communicationprotocol.h>

namespace BC::Key {
static const QString customComm("comm");
static const QString customType("type");
static const QString customKey("key");
static const QString customInt("int");
static const QString customIntMin("min");
static const QString customIntMax("max");
static const QString customString("string");
static const QString customStringMaxLength("length");
static const QString customTypeLabel("name");
}

class CustomInstrument : public CommunicationProtocol
{
public:
    explicit CustomInstrument(QString key, QObject *parent = nullptr);

public slots:
    void initialize() override;
    bool testConnection() override;
};

#endif // CUSTOMINSTRUMENT_H
