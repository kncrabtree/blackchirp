#ifndef IOBOARDCONFIGWIDGET_H
#define IOBOARDCONFIGWIDGET_H

#include <gui/widget/digitizerconfigwidget.h>
#include <hardware/optional/ioboard/ioboardconfig.h>

namespace BC::Key::DigiWidget {
static const QString channelName("name");
}

class QTableWidget;

class IOBoardConfigWidget : public DigitizerConfigWidget
{
    Q_OBJECT
public:
    explicit IOBoardConfigWidget(QWidget *parent = nullptr);
    ~IOBoardConfigWidget();

    void setFromConfig(const IOBoardConfig &cfg);
    void toConfig(IOBoardConfig &cfg);

    std::map<int,QString> getAnalogNames() const;
    std::map<int,QString> getDigitalNames() const;

signals:

private:
    QTableWidget *p_analogNameWidget{nullptr}, *p_digitalNameWidget{nullptr};

};

#endif // IOBOARDCONFIGWIDGET_H
