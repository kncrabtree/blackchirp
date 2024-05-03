#include <gui/dialog/quickexptdialog.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QSpacerItem>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QGroupBox>

#include <data/experiment/experiment.h>
#include <data/storage/settingsstorage.h>
#include <gui/widget/experimentsummarywidget.h>

#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/ioboard/ioboard.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

QuickExptDialog::QuickExptDialog(const std::map<QString, QString> &hwl, QWidget *parent) :
    QDialog(parent), d_hardware{hwl}
{
    setWindowTitle("Quick Experiment");
    auto vbl = new QVBoxLayout;

    auto tophbl = new QHBoxLayout;
    auto egb = new QGroupBox("Experiment");
    auto gl = new QGridLayout;

    gl->addWidget(new QLabel("Number"),0,0);

    SettingsStorage s;
    int expNum = s.get(BC::Key::exptNum,0);
    p_expSpinBox = new QSpinBox;
    if(expNum < 1)
    {
        p_expSpinBox->setRange(0,0);
        p_expSpinBox->setSpecialValueText("N/A");
    }
    p_expSpinBox->setRange(1,expNum);
    connect(p_expSpinBox,qOverload<int>(&QSpinBox::valueChanged),this,&QuickExptDialog::loadExperiment);

    gl->addWidget(p_expSpinBox,0,1);

    p_warningLabel = new QLabel;
    p_warningLabel->setWordWrap(true);
    p_warningLabel->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
    p_warningLabel->setMinimumSize({0,60});
    gl->addWidget(p_warningLabel,1,0,1,2);

    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);
    gl->setRowStretch(0,0);
    gl->setRowStretch(1,1);
    egb->setLayout(gl);
    tophbl->addWidget(egb,1);

    auto hwgb = new QGroupBox("Use Current Settings");
    hwgb->setToolTip("If checked, the experiment will use the current settings\nrather than the saved settings for this hardware item.");

    p_hwLayout = new QFormLayout;
    hwgb->setLayout(p_hwLayout);
    tophbl->addWidget(hwgb,1);
    vbl->addLayout(tophbl,0);

    p_esw = new ExperimentSummaryWidget;
    vbl->addWidget(p_esw,1);

    auto bl = new QHBoxLayout;
    auto ne = new QPushButton("New Experiment");
    p_cfgButton = new QPushButton("Configure Experiment");
    p_startButton = new QPushButton("Start Experiment");
    auto cb = new QPushButton("Cancel");

    p_cfgButton->setEnabled(false);
    p_startButton->setEnabled(false);

    bl->addWidget(ne);
    bl->addWidget(p_cfgButton);
    bl->addItem(new QSpacerItem(0,0,QSizePolicy::Expanding,QSizePolicy::Fixed));
    bl->addWidget(p_startButton);
    bl->addWidget(cb);

    connect(ne,&QPushButton::clicked,[this](){ done(New); });
    connect(p_cfgButton,&QPushButton::clicked,[this](){ done(Configure); });
    connect(p_startButton,&QPushButton::clicked,[this](){ done(Start); });
    connect(cb,&QPushButton::clicked,this,&QuickExptDialog::reject);

    vbl->addLayout(bl);

    setLayout(vbl);

    std::set<QString> optHw{ BC::Key::PController::key, BC::Key::Flow::flowController, BC::Key::PGen::key, BC::Key::IOB::ioboard, BC::Key::TC::key};

    for(auto &[key,subKey] : hwl)
    {
        auto ki = BC::Key::parseKey(key);
        auto hwType = ki.first;

        auto it = optHw.find(hwType);
        if(it != optHw.end())
        {
            SettingsStorage s(key,SettingsStorage::Hardware);
            auto cb = new QCheckBox;
            cb->setChecked(true);
            auto lbl = new QLabel(s.get(BC::Key::HW::name,*it));

            p_hwLayout->addRow(lbl,cb);
            d_hwBoxes.insert({key,cb});
        }
    }

    p_expSpinBox->setValue(p_expSpinBox->maximum());
}

std::map<QString, bool> QuickExptDialog::getOptHwSettings() const
{
    std::map<QString,bool> out;
    for(const auto &[key,cb] : d_hwBoxes)
        out.emplace(key,cb->isChecked());

    return out;
}

int QuickExptDialog::exptNumber() const
{
    return p_expSpinBox->value();
}

void QuickExptDialog::loadExperiment(int num)
{
    Experiment exp(num,"",true);
    p_esw->setExperiment(&exp);

    bool hwIdentical = true;
    if(d_hardware.size() != exp.d_hardware.size())
        hwIdentical = false;
    else
    {
        for(auto const &[key,val] : d_hardware)
        {
            auto it = exp.d_hardware.find(key);
            if(it == exp.d_hardware.end() || it->second != val)
            {
                hwIdentical = false;
                break;
            }
        }
    }

    p_cfgButton->setEnabled(hwIdentical);
    p_startButton->setEnabled(hwIdentical);

    if(!hwIdentical)
    {
        p_warningLabel->setText(QString("Error: Cannot repeat experiment %1 because the current hardware configuration is different.").arg(p_expSpinBox->value()));
        p_warningLabel->setStyleSheet("QLabel { color : red; font-weight : bold; }");
        return;
    }

    if(exp.d_majorVersion != QString(STRINGIFY(BC_MAJOR_VERSION)))
    {
        p_warningLabel->setText(QString("Error: Cannot repeat experiment %1 because it was recorded with a different major version of Blackchirp.").arg(p_expSpinBox->value()));
        p_warningLabel->setStyleSheet("QLabel { color : red; font-weight : bold; }");
        p_cfgButton->setEnabled(false);
        p_startButton->setEnabled(false);
        return;
    }
    else if(exp.d_minorVersion != QString(STRINGIFY(BC_MINOR_VERSION)))
    {
        p_warningLabel->setText(QString("Warning: Experiment %1 was recorded with a different minor version of Blackchirp. Some settings may not work correctly.\n\nIt is strongly recommended that you configure this experiment manually.").arg(p_expSpinBox->value()));
        p_warningLabel->setStyleSheet("QLabel { font-weight : bold; }");
        return;
    }

    p_warningLabel->clear();
}
