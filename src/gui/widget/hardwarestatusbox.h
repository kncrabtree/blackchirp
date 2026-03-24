#ifndef HARDWARESTATUSBOX_H
#define HARDWARESTATUSBOX_H

#include <QGroupBox>

class HardwareStatusBox : public QGroupBox
{
    Q_OBJECT
public:
    HardwareStatusBox(QString key, QWidget *parent = nullptr);

protected:
    QString d_key;
    
    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // HARDWARESTATUSBOX_H
