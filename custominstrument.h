#ifndef CUSTOMINSTRUMENT_H
#define CUSTOMINSTRUMENT_H

#include "communicationprotocol.h"

class CustomInstrument : public CommunicationProtocol
{
public:
    explicit CustomInstrument(QString key, QString subKey, QObject *parent = nullptr);

public slots:
    void initialize();
    bool testConnection();
};

#endif // CUSTOMINSTRUMENT_H
