#ifndef VIRTUALINSTRUMENT_H
#define VIRTUALINSTRUMENT_H

#include "communicationprotocol.h"

class VirtualInstrument : public CommunicationProtocol
{
    Q_OBJECT
public:
    explicit VirtualInstrument(QString key, QObject *parent = nullptr);
    ~VirtualInstrument();

public slots:
    void initialize() override;
    bool testConnection() override;
};

#endif // VIRTUALINSTRUMENT_H
