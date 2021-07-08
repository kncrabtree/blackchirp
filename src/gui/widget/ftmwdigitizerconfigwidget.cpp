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
    if(p_blockAverageBox->isChecked())
        p_numAveragesBox->setValue(numChirps);
    else if(p_multiRecordBox->isChecked())
        p_numRecordsBox->setValue(numChirps);
    else
    {
        p_numAveragesBox->setValue(numChirps);
        p_numRecordsBox->setValue(numChirps);
    }
}
