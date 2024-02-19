#include "ftmwdigitizerconfigwidget.h"

#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <QCheckBox>
#include <QSpinBox>

FtmwDigitizerConfigWidget::FtmwDigitizerConfigWidget(QWidget *parent) :
    DigitizerConfigWidget("FtmwDigitizerConfigWidget",BC::Key::FtmwScope::ftmwScope,parent)
{

}

void FtmwDigitizerConfigWidget::configureForChirp(int numChirps)
{
    p_numRecordsBox->setValue(numChirps);
}
