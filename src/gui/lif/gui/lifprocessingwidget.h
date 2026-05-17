#ifndef LIFPROCESSINGWIDGET_H
#define LIFPROCESSINGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/lif/liftrace.h>

namespace BC::Key::LifProcessing {
const QString key("lifProcessingWidget");
const QString lgStart("lifGateStart");
const QString lgEnd("lifGateEnd");
const QString rgStart("refGateStart");
const QString rgEnd("refGateEnd");
const QString lpAlpha("lowPassAlpha");
const QString sgEn("savGolEnabled");
const QString sgWin("savGolWindow");
const QString sgPoly("savGolPoly");
}

class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QToolButton;
class QHBoxLayout;
class QResizeEvent;

class LifProcessingWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit LifProcessingWidget(bool store = true, QWidget *parent = nullptr);

    void initialize(int recLen, bool ref);
    void setAll(const LifTrace::LifProcSettings &lc);
    LifTrace::LifProcSettings getSettings() const;

    void experimentComplete();

protected:
    void resizeEvent(QResizeEvent *event) override;

signals:
    void settingChanged();
    void reprocessSignal();
    void resetSignal();
    void saveSignal();

private:
    void adjustButtonStyle();

    QSpinBox *p_lgStartBox, *p_lgEndBox, *p_rgStartBox, *p_rgEndBox, *p_sgWinBox, *p_sgPolyBox;
    QDoubleSpinBox *p_lpAlphaBox;
    QCheckBox *p_sgBox;
    QToolButton *p_reprocessButton, *p_saveButton, *p_resetButton;
    QHBoxLayout *p_btnLayout;
};

#endif // LIFPROCESSINGWIDGET_H
