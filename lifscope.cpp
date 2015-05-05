#include "lifscope.h"

LifScope::LifScope(QObject *parent) :
    HardwareObject(parent)
{
    d_key = QString("lifScope");

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    d_refEnabled = s.value(QString("%1/refEnabled").arg(d_key),false).toBool();

}

LifScope::~LifScope()
{

}

void LifScope::setAll(const BlackChirp::LifScopeConfig c)
{
    if(!qFuzzyCompare(d_config.vScale1,c.vScale1))
        setLifVScale(c.vScale1);

    if(c.recordLength != d_config.recordLength || !qFuzzyCompare(c.sampleRate,d_config.sampleRate))
        setHorizontalConfig(c.sampleRate,c.recordLength);

    if(d_config.refEnabled != c.refEnabled)
        setRefEnabled(c.refEnabled);

    if(d_config.refEnabled && !qFuzzyCompare(d_config.vScale2,c.vScale2))
        setRefVScale(c.vScale2);

    emit configUpdated(d_config);
}

