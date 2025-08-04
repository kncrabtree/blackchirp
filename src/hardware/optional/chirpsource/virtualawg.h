#ifndef VIRTUALAWG_H
#define VIRTUALAWG_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC::Key::AWG {
static const QString virtualAwg{"virtualAwg"};
static const QString virtualAwgName("Virtual Arbitrary Waveform Generator");
}

class VirtualAwg : public AWG
{
    Q_OBJECT
public:
    explicit VirtualAwg(const QString& label, QObject *parent = nullptr);
    ~VirtualAwg();

protected:
    bool testConnection() override;
    void initialize() override;

};


#endif // VIRTUALAWG_H
