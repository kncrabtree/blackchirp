#ifndef HARDWARESTATUSBOX_H
#define HARDWARESTATUSBOX_H

#include <QFrame>

class QLabel;
class QToolButton;
class QWidget;

class HardwareStatusBox : public QFrame
{
    Q_OBJECT
public:
    HardwareStatusBox(const QString &key, QWidget *parent = nullptr);

signals:
    void configureRequested();

protected:
    QString d_key;
    QWidget *body() const;

    // QWidget interface
public:
    QSize sizeHint() const override;

private:
    QWidget *p_body;
};

#endif // HARDWARESTATUSBOX_H
