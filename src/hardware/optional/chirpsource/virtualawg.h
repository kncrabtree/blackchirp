#ifndef VIRTUALAWG_H
#define VIRTUALAWG_H

#include <hardware/optional/chirpsource/awg.h>

namespace BC {
namespace Key {
static const QString vawgName("Virtual Arbitrary Waveform Generator");
}
}

class VirtualAwg : public AWG
{
    Q_OBJECT
public:
    explicit VirtualAwg(QObject *parent = nullptr);
    ~VirtualAwg();

protected:
    bool testConnection() override;
    void initialize() override;

};


#endif // VIRTUALAWG_H
