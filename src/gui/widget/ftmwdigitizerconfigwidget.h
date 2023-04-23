#ifndef FTMWDIGITIZERCONFIGWIDGET_H
#define FTMWDIGITIZERCONFIGWIDGET_H

#include <gui/widget/digitizerconfigwidget.h>

#include <data/experiment/chirpconfig.h>

class FtmwDigitizerConfigWidget : public DigitizerConfigWidget
{
    Q_OBJECT
public:
    FtmwDigitizerConfigWidget(QWidget *parent = nullptr);
    ~FtmwDigitizerConfigWidget(){}

    void configureForChirp(int numChirps);
};

#endif // FTMWDIGITIZERCONFIGWIDGET_H
