#ifndef HARDWARESTATUSBOX_H
#define HARDWARESTATUSBOX_H

#include <QGroupBox>

class HardwareStatusBox : public QGroupBox
{
    Q_OBJECT
public:
    HardwareStatusBox(QString key, QWidget *parent = nullptr);
    const QString d_titleTemplate{"%1 Status"};

public slots:
    void updateTitle(const QString &n);

protected:
    QString d_key;
};

#endif // HARDWARESTATUSBOX_H
