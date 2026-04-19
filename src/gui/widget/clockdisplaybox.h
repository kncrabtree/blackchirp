#ifndef CLOCKDISPLAYBOX_H
#define CLOCKDISPLAYBOX_H

#include <QHash>

#include <data/experiment/rfconfig.h>
#include <gui/widget/hardwarestatusbox.h>

class QLabel;
class QToolButton;

class ClockDisplayBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    explicit ClockDisplayBox(QWidget *parent = nullptr);

signals:
    void clockHardwareRequested(const QString &hwKey);

public slots:
    void updateFrequency(RfConfig::ClockType t, double f);
    void setClockHardware(RfConfig::ClockType type, const QString &hwKey, int output);

private:
    struct ClockRow {
        QLabel *nameLabel{nullptr};
        QLabel *valueLabel{nullptr};
        QToolButton *cogButton{nullptr};
        QString hwKey;
    };

    QHash<RfConfig::ClockType, ClockRow> d_rows;
    int d_decimals{4};
};

#endif // CLOCKDISPLAYBOX_H
