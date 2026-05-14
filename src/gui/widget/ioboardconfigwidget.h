#ifndef IOBOARDCONFIGWIDGET_H
#define IOBOARDCONFIGWIDGET_H

#include <gui/widget/digitizerconfigwidget.h>
#include <data/experiment/hardware/optional/ioboard/ioboardconfig.h>

class IOBoardConfigWidget : public DigitizerConfigWidget
{
    Q_OBJECT
public:
    explicit IOBoardConfigWidget(IOBoardConfig &cfg, QWidget *parent = nullptr);
    ~IOBoardConfigWidget();

    void setFromConfig(const IOBoardConfig &cfg);
    void toConfig(IOBoardConfig &cfg);

    std::map<int,QString> getAnalogNames() const;
    std::map<int,QString> getDigitalNames() const;

signals:

private:
    QString d_key;

};

#endif // IOBOARDCONFIGWIDGET_H
