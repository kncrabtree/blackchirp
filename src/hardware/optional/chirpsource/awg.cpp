#include <hardware/optional/chirpsource/awg.h>

AWG::AWG(const QString& impl, const QString& label, QObject *parent) :
    HardwareObject(QString(AWG::staticMetaObject.className()), impl, label, parent)
{

}

AWG::~AWG()
{

}
