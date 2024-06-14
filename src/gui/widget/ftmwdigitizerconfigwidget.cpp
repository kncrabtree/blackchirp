#include "ftmwdigitizerconfigwidget.h"

#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <QCheckBox>
#include <QSpinBox>
#include <QGroupBox>

FtmwDigitizerConfigWidget::FtmwDigitizerConfigWidget(QWidget *parent) :
    DigitizerConfigWidget("FtmwDigitizerConfigWidget",BC::Key::hwKey(BC::Key::FtmwScope::ftmwScope,0),parent)
{
}

void FtmwDigitizerConfigWidget::configureForChirp(int numChirps)
{
    p_numRecordsBox->setValue(numChirps);
}
