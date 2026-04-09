#ifndef PYTHONHARDWARECONTROLWIDGET_H
#define PYTHONHARDWARECONTROLWIDGET_H

#include <QWidget>

class QLabel;
class HardwareManager;

/*!
 * \brief Control widget for Python hardware implementations
 *
 * Provides script path display, "Open in Editor" and "Reload Script" buttons,
 * and a status label. Intended for use in HWDialog for Python hardware objects.
 */
class PythonHardwareControlWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PythonHardwareControlWidget(const QString &hwKey, HardwareManager *hwm, QWidget *parent = nullptr);

private slots:
    void onReloadResult(const QString &hwKey, bool success, const QString &msg);

private:
    QString d_hwKey;
    HardwareManager *p_hwm;
    QLabel *p_statusLabel;
};

#endif // PYTHONHARDWARECONTROLWIDGET_H
