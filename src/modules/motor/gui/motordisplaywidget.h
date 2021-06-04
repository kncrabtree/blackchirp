#ifndef MOTORDISPLAYWIDGET_H
#define MOTORDISPLAYWIDGET_H

#include <QWidget>

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QAction>
#include <QtWidgets/QMenu>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QWidgetAction>
#include <QtWidgets/QToolButton>

#include <src/modules/motor/gui/motorsliderwidget.h>
#include <src/modules/motor/gui/motortimeplot.h>
#include <src/modules/motor/gui/motorxyplot.h>
#include <src/modules/motor/gui/motorzplot.h>

#include <src/data/experiment/experiment.h>
#include <src/modules/motor/data/motorscan.h>
#include <src/data/analysis/analysis.h>

class MotorSliderWidget;

namespace Ui {
class MotorDisplayWidget;
}

class MotorDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MotorDisplayWidget(QWidget *parent = 0);
    ~MotorDisplayWidget();

public slots:
    void prepareForScan(const MotorScan s);
    void newMotorData(const MotorScan s);
    void updatePlots();
    void smoothBoxChecked(bool checked);
    void updateCoefs();
    void winSizeChanged(int w);
    void polySizeChanged(int p);

private:
    Ui::MotorDisplayWidget *ui;
    QList<MotorSliderWidget*> d_sliders;

    MotorScan d_currentScan;

    int d_winSize, d_polyOrder;
    bool d_smooth;
    Eigen::MatrixXd d_coefs;

};



QT_BEGIN_NAMESPACE

class Ui_MotorDisplayWidget
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    MotorZPlot *motorZSpectrogramPlot;
    MotorSliderWidget *zSlider1;
    MotorSliderWidget *zSlider2;
    QHBoxLayout *horizontalLayout_2;
    MotorXYPlot *motorXYSpectrogramPlot;
    MotorSliderWidget *xySlider1;
    MotorSliderWidget *xySlider2;
    MotorTimePlot *motorTimePlot;
    MotorSliderWidget *timeXSlider;
    MotorSliderWidget *timeYSlider;
    MotorSliderWidget *timeZSlider;
    QCheckBox *smoothBox;
    QSpinBox *winBox;
    QSpinBox *polyBox;
    QToolBar *toolBar;


    void setupUi(QWidget *MotorDisplayWidget)
    {
        if (MotorDisplayWidget->objectName().isEmpty())
            MotorDisplayWidget->setObjectName(QStringLiteral("MotorDisplayWidget"));
        MotorDisplayWidget->resize(843, 504);
        verticalLayout = new QVBoxLayout(MotorDisplayWidget);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));

        toolBar = new QToolBar;
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);


        auto settingsAction = toolBar->addAction(QIcon(":/icons/configure.png"),QString("Settings"));
        auto settingsButton = dynamic_cast<QToolButton*>(toolBar->widgetForAction(settingsAction));
        auto settingsMenu = new QMenu;
        auto settingsWa = new QWidgetAction(settingsMenu);
        auto settingsWidget = new QWidget;
        auto settingsFl = new QFormLayout;

        smoothBox = new QCheckBox;
        smoothBox->setEnabled(false);
        auto smoothLbl = new QLabel(QString("Smoothing"));
        smoothLbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        settingsFl->addRow(smoothLbl,smoothBox);

        winBox = new QSpinBox;
        winBox->setRange(5,100000);
        winBox->setEnabled(false);
        winBox->setSingleStep(2);
        winBox->setKeyboardTracking(false);
        winBox->setToolTip(QString("Window size for Savitzky-Golay smoothing. Must be odd."));
        auto winLbl = new QLabel(QString("Window Size"));
        winLbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        settingsFl->addRow(winLbl,winBox);

        polyBox = new QSpinBox;
        polyBox->setRange(2,11);
        polyBox->setEnabled(false);
        polyBox->setSingleStep(1);
        polyBox->setKeyboardTracking(false);
        polyBox->setToolTip(QString("Polynomial order for Savitzky-Golay smoothing. Must be less than window size."));
        auto polyLbl = new QLabel(QString("Polynomial order"));
        polyLbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        settingsFl->addRow(polyLbl,polyBox);

        settingsWidget->setLayout(settingsFl);
        settingsWa->setDefaultWidget(settingsWidget);
        settingsMenu->addAction(settingsWa);
        settingsButton->setMenu(settingsMenu);
        settingsButton->setPopupMode(QToolButton::InstantPopup);

        verticalLayout->addWidget(toolBar,0,Qt::AlignLeft);


        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        motorZSpectrogramPlot = new MotorZPlot(MotorDisplayWidget);
        motorZSpectrogramPlot->setObjectName(QStringLiteral("motorZSpectrogramPlot"));

        horizontalLayout->addWidget(motorZSpectrogramPlot);

        zSlider1 = new MotorSliderWidget(MotorDisplayWidget);
        zSlider1->setObjectName(QStringLiteral("zSlider1"));

        horizontalLayout->addWidget(zSlider1);

        zSlider2 = new MotorSliderWidget(MotorDisplayWidget);
        zSlider2->setObjectName(QStringLiteral("zSlider2"));

        horizontalLayout->addWidget(zSlider2);

        horizontalLayout->setStretch(0, 1);

        verticalLayout->addLayout(horizontalLayout,1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        motorXYSpectrogramPlot = new MotorXYPlot(MotorDisplayWidget);
        motorXYSpectrogramPlot->setObjectName(QStringLiteral("motorXYSpectrogramPlot"));

        horizontalLayout_2->addWidget(motorXYSpectrogramPlot);

        xySlider1 = new MotorSliderWidget(MotorDisplayWidget);
        xySlider1->setObjectName(QStringLiteral("xySlider1"));

        horizontalLayout_2->addWidget(xySlider1);

        xySlider2 = new MotorSliderWidget(MotorDisplayWidget);
        xySlider2->setObjectName(QStringLiteral("xySlider2"));

        horizontalLayout_2->addWidget(xySlider2);

        motorTimePlot = new MotorTimePlot(MotorDisplayWidget);
        motorTimePlot->setObjectName(QStringLiteral("motorTimePlot"));

        horizontalLayout_2->addWidget(motorTimePlot);

        timeXSlider = new MotorSliderWidget(MotorDisplayWidget);
        timeXSlider->setObjectName(QStringLiteral("timeXSlider"));

        horizontalLayout_2->addWidget(timeXSlider);

        timeYSlider = new MotorSliderWidget(MotorDisplayWidget);
        timeYSlider->setObjectName(QStringLiteral("timeYSlider"));

        horizontalLayout_2->addWidget(timeYSlider);

        timeZSlider = new MotorSliderWidget(MotorDisplayWidget);
        timeZSlider->setObjectName(QStringLiteral("timeZSlider"));

        horizontalLayout_2->addWidget(timeZSlider);

        horizontalLayout_2->setStretch(0, 1);
        horizontalLayout_2->setStretch(3, 1);

        verticalLayout->addLayout(horizontalLayout_2,1);


        retranslateUi(MotorDisplayWidget);

        QMetaObject::connectSlotsByName(MotorDisplayWidget);
    } // setupUi

    void retranslateUi(QWidget *MotorDisplayWidget)
    {
        MotorDisplayWidget->setWindowTitle(QApplication::translate("MotorDisplayWidget", "Form", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MotorDisplayWidget: public Ui_MotorDisplayWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MOTORDISPLAYWIDGET_H
