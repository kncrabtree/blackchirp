#ifndef CUSTOMINSTRUMENT_H
#define CUSTOMINSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>

namespace BC::Key::Custom {
static const QString comm{"comm"};
static const QString type{"type"};
static const QString key{"key"};
static const QString intKey{"int"};
static const QString intMin{"min"};
static const QString intMax{"max"};
static const QString stringKey{"string"};
static const QString maxLen{"length"};
static const QString label{"name"};
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
