#include "ftmwdigitizerconfigwidget.h"

#include <hardware/core/ftmwdigitizer/ftmwdigitizer.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <QCheckBox>
#include <QSpinBox>
#include <QGroupBox>

FtmwDigitizerConfigWidget::FtmwDigitizerConfigWidget(QWidget *parent) :
    DigitizerConfigWidget("FtmwDigitizerConfigWidget",
                          RuntimeHardwareConfig::constInstance().getActiveKeys<FtmwDigitizer>().value(0),
                          false,
                          parent)
{
}

void FtmwDigitizerConfigWidget::configureForChirp(int numChirps)
{
    p_numRecordsBox->setValue(numChirps);
}
