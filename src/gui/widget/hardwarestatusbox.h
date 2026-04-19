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
    void setTitle(const QString &t);
    void setConfigButtonTooltip(const QString &t);

private:
    QWidget *p_body;
    QLabel *p_titleLabel{nullptr};
    QToolButton *p_configButton{nullptr};
};

#endif // HARDWARESTATUSBOX_H
