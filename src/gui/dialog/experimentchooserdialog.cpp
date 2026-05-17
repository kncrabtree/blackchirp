#include "experimentchooserdialog.h"

#include <climits>

#include <QVBoxLayout>
#include <QSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QToolButton>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QFileDialog>
#include <QFontMetrics>
#include <QScreen>
#include <QStyle>
#include <QDir>

#include <gui/style/themecolors.h>
#include <gui/util/recentexperiments.h>
#include <gui/widget/settingstable.h>
#include <data/storage/blackchirpcsv.h>

ExperimentChooserDialog::ExperimentChooserDialog(int upperBound,
                                                 const QString &browseStartDir,
                                                 const std::vector<RecentEntry> &recent,
                                                 QWidget *parent) :
    QDialog(parent),
    d_browseDir(browseStartDir),
    d_recent(recent)
{
    setWindowTitle(QString("Open experiment"));
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/document-plus.svg",
                                                ThemeColors::IconPrimary, this));

    // No selectable experiment number forces custom-path mode: the
    // "Specify custom path" section starts checked (and the number
    // control is disabled).
    const bool forcePath = upperBound < 1;

    p_numBox = new QSpinBox(this);
    p_pathEdit = new QLineEdit(this);
    p_browseButton = new QToolButton(this);
    p_browseButton->setIcon(ThemeColors::createThemedIcon(
        ":/icons/document-magnifying-glass.svg", ThemeColors::IconSecondary, this));

    auto *table = new SettingsTable(this);

    // Recent experiments, newest first. Selecting an entry fills the
    // number / path controls; the user still presses Open to confirm.
    if(!d_recent.empty())
    {
        p_recentBox = new QComboBox(this);
        p_recentBox->addItem(QString("Select a recent experiment..."));
        for(const auto &e : d_recent)
            p_recentBox->addItem(BC::RecentExperiments::displayText(e.num, e.path));

        connect(p_recentBox, qOverload<int>(&QComboBox::activated), this,
                [this](int idx){
            if(idx < 1)
                return;
            const auto &e = d_recent[idx-1];
            if(e.path.isEmpty())
            {
                p_pathBox->setChecked(false);
                p_numBox->setValue(e.num);
            }
            else
            {
                p_pathBox->setChecked(true);
                p_pathEdit->setText(e.path);
            }
        });
        table->addSettingRow(QString("Recent"), p_recentBox);
    }

    if(forcePath)
    {
        p_numBox->setRange(0, INT_MAX);
        p_numBox->setSpecialValueText(QString("Select..."));
        p_numBox->setEnabled(false);
    }
    else
    {
        p_numBox->setRange(1, upperBound);
        p_numBox->setValue(upperBound);
    }
    table->addSettingRow(QString("Experiment Number"), p_numBox);

    const int pathSection = table->addCheckableSectionRow(
        QString("Specify custom path"), forcePath, &p_pathBox);
    const int pathRow = table->addSettingRow(QString("Directory"),
                                             p_pathEdit, p_browseButton);
    // Unchecking the section collapses the directory row; checking it
    // expands it and disables the experiment-number control.
    table->bindSectionRows(pathSection, {pathRow});
    connect(p_pathBox, &QCheckBox::toggled, this, [this](bool checked){
        p_numBox->setEnabled(!checked);
        if(!checked)
            p_pathEdit->clear();
    });
    p_numBox->setEnabled(!forcePath);

    connect(p_browseButton, &QToolButton::clicked, this, [this](){
        QString startDir = d_browseDir.isEmpty() ? QDir::homePath() : d_browseDir;
        QString path = QFileDialog::getExistingDirectory(
            this, QString("Select experiment directory"), startDir);
        if(!path.isEmpty())
        {
            p_pathEdit->setText(path);
            d_browseDir = path;
            emit browseDirChanged(path);
        }
    });

    auto *bb = new QDialogButtonBox(QDialogButtonBox::Open|QDialogButtonBox::Cancel,
                                    this);
    connect(bb, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto *vbl = new QVBoxLayout;
    vbl->addWidget(table);
    vbl->addWidget(bb);
    setLayout(vbl);

    // A directory path is the widest thing this dialog shows. Size it
    // from font metrics (valid here, unlike child sizeHints before the
    // dialog is realized): a ~50-character value-column floor, widened
    // if the recent combo's longest entry needs more, plus the label
    // column. Capped so a pathological path cannot exceed the screen.
    const QFontMetrics fm(font());
    const int charW = qMax(1, fm.averageCharWidth());

    int valueW = charW * 50;
    if(p_recentBox)
    {
        p_recentBox->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        int longest = 0;
        for(int i = 0; i < p_recentBox->count(); ++i)
            longest = qMax(longest,
                           fm.horizontalAdvance(p_recentBox->itemText(i)));
        const int comboChrome =
            2 * style()->pixelMetric(QStyle::PM_ComboBoxFrameWidth)
            + style()->pixelMetric(QStyle::PM_MenuButtonIndicator) + 8;
        valueW = qMax(valueW, longest + comboChrome);
    }

    int labelW = 0;
    for(const auto &l : {QString("Recent"), QString("Experiment Number"),
                         QString("Directory")})
        labelW = qMax(labelW, fm.horizontalAdvance(l));
    labelW += 16; // cell horizontal padding

    // The "Specify custom path" band spans both columns; keep the
    // dialog wide enough for it too.
    const int sectionW = fm.horizontalAdvance(QString("Specify custom path"))
                         + 32;

    const QMargins m = layout()->contentsMargins();
    const int chrome = m.left() + m.right() + 8;

    int targetW = qMax(labelW + valueW, sectionW) + chrome;
    if(const QScreen *s = screen())
        targetW = qMin(targetW, qRound(s->availableGeometry().width() * 0.9));

    setMinimumWidth(targetW);
    resize(targetW, sizeHint().height());
}

void ExperimentChooserDialog::accept()
{
    if(p_pathBox->isChecked())
    {
        const QString path = p_pathEdit->text();
        if(path.isEmpty())
        {
            QMessageBox::critical(this, QString("Load error"),
                QString("Cannot open experiment with an empty path."),
                QMessageBox::Ok);
            return;
        }

        QDir dir(path);
        if(!dir.exists())
        {
            QMessageBox::critical(this, QString("Load error"),
                QString("The directory %1 does not exist. Could not load experiment.")
                    .arg(dir.absolutePath()), QMessageBox::Ok);
            return;
        }

        d_resultNum = 0;
        d_resultPath = path;
        QDialog::accept();
        return;
    }

    const int num = p_numBox->value();
    if(num < 1)
    {
        QMessageBox::critical(this, QString("Load error"),
            QString("Cannot open an experiment numbered below 1. (You chose %1)")
                .arg(num), QMessageBox::Ok);
        return;
    }
    if(!BlackchirpCSV::exptDirExists(num))
    {
        QMessageBox::critical(this, QString("Load error"),
            QString("No experiment numbered %1 was found in the active data path. "
                    "Verify the number or use \"Specify custom path\" to select a "
                    "directory directly.").arg(num), QMessageBox::Ok);
        return;
    }

    d_resultNum = num;
    d_resultPath.clear();
    QDialog::accept();
}
