#include "pythonsettingswidget.h"

#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QProcess>
#include <QSignalBlocker>

#include <hardware/python/pythonhardwarebase.h>
#include <gui/style/themecolors.h>

PythonSettingsWidget::PythonSettingsWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    // Script path row
    auto *scriptRow = new QWidget(this);
    auto *scriptLayout = new QHBoxLayout(scriptRow);
    scriptLayout->setContentsMargins(0, 0, 0, 0);

    auto *scriptLabel = new QLabel(tr("Python Script:"), scriptRow);
    p_scriptEdit = new QLineEdit(scriptRow);
    p_scriptEdit->setPlaceholderText(tr("Path to Python hardware script..."));

    auto *browseButton = new QPushButton(tr("Browse..."), scriptRow);
    connect(browseButton, &QPushButton::clicked, this, [this]() {
        QString path = QFileDialog::getOpenFileName(this, tr("Select Python Script"),
                                                    QString(), tr("Python Files (*.py)"));
        if (!path.isEmpty())
            p_scriptEdit->setText(path);
    });

    scriptLayout->addWidget(scriptLabel);
    scriptLayout->addWidget(p_scriptEdit, 1);
    scriptLayout->addWidget(browseButton);
    mainLayout->addWidget(scriptRow);

    // Class name row
    auto *classRow = new QWidget(this);
    auto *classLayout = new QHBoxLayout(classRow);
    classLayout->setContentsMargins(0, 0, 0, 0);

    auto *classLabel = new QLabel(tr("Python Class:"), classRow);
    p_classCombo = new QComboBox(classRow);
    p_classCombo->setEditable(true);

    classLayout->addWidget(classLabel);
    classLayout->addWidget(p_classCombo, 1);
    mainLayout->addWidget(classRow);

    // Environment path row
    auto *envRow = new QWidget(this);
    auto *envLayout = new QHBoxLayout(envRow);
    envLayout->setContentsMargins(0, 0, 0, 0);

    auto *envLabel = new QLabel(tr("Python Environment:"), envRow);
    p_envEdit = new QLineEdit(envRow);
    p_envEdit->setPlaceholderText(tr("Path to venv or conda environment (leave empty for system Python)..."));
    p_envEdit->setToolTip(tr("Path to a venv or conda environment directory. Leave empty to use the system Python."));

    auto *envBrowseButton = new QPushButton(tr("Browse..."), envRow);
    connect(envBrowseButton, &QPushButton::clicked, this, [this]() {
        QString path = QFileDialog::getExistingDirectory(this, tr("Select Python Environment Directory"));
        if (!path.isEmpty())
            p_envEdit->setText(path);
    });

    envLayout->addWidget(envLabel);
    envLayout->addWidget(p_envEdit, 1);
    envLayout->addWidget(envBrowseButton);
    mainLayout->addWidget(envRow);

    // Environment status label
    p_envStatusLabel = new QLabel(this);
    mainLayout->addWidget(p_envStatusLabel);

    // Debounce timer: update status 500 ms after the user stops typing
    p_envStatusTimer = new QTimer(this);
    p_envStatusTimer->setSingleShot(true);
    connect(p_envStatusTimer, &QTimer::timeout, this, &PythonSettingsWidget::updateEnvStatus);

    // Signal connections
    connect(p_scriptEdit, &QLineEdit::textChanged, this, [this](const QString &text) {
        populateClassCombo(text);
        emit scriptPathChanged(text);
    });

    connect(p_classCombo, &QComboBox::currentTextChanged, this,
            &PythonSettingsWidget::classNameChanged);

    connect(p_envEdit, &QLineEdit::textChanged, this, [this](const QString &text) {
        p_envStatusTimer->start(500);
        emit envPathChanged(text);
    });
}

QString PythonSettingsWidget::scriptPath() const
{
    return p_scriptEdit->text();
}

void PythonSettingsWidget::setScriptPath(const QString &path)
{
    QSignalBlocker blocker(p_scriptEdit);
    p_scriptEdit->setText(path);
    populateClassCombo(path);
}

QString PythonSettingsWidget::className() const
{
    return p_classCombo->currentText();
}

void PythonSettingsWidget::setClassName(const QString &name)
{
    QSignalBlocker blocker(p_classCombo);
    p_classCombo->setCurrentText(name);
}

void PythonSettingsWidget::setClassNamePlaceholder(const QString &placeholder)
{
    p_classCombo->lineEdit()->setPlaceholderText(placeholder);
}

QString PythonSettingsWidget::envPath() const
{
    return p_envEdit->text();
}

void PythonSettingsWidget::setEnvPath(const QString &path)
{
    QSignalBlocker blocker(p_envEdit);
    p_envEdit->setText(path);
    updateEnvStatus();
}

void PythonSettingsWidget::populateClassCombo(const QString &scriptPath)
{
    QString current = p_classCombo->currentText();
    {
        QSignalBlocker blocker(p_classCombo);
        p_classCombo->clear();
    }
    if (!scriptPath.isEmpty()) {
        QFile file(scriptPath);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&file);
            QRegularExpression re(QStringLiteral("^class\\s+(\\w+)"));
            while (!in.atEnd()) {
                auto match = re.match(in.readLine());
                if (match.hasMatch())
                    p_classCombo->addItem(match.captured(1));
            }
        }
    }
    if (!current.isEmpty())
        p_classCombo->setCurrentText(current);
}

void PythonSettingsWidget::updateEnvStatus()
{
    const QString envPath = p_envEdit->text().trimmed();
    const QString exe = PythonHardwareBase::resolvePythonExecutable(envPath);
    const QString version = getPythonVersion(exe);

    if (envPath.isEmpty()) {
        if (!version.isEmpty()) {
            p_envStatusLabel->setText(tr("System: %1").arg(version));
            p_envStatusLabel->setStyleSheet(
                QString("color: %1;").arg(ThemeColors::getCSSColor(
                    ThemeColors::SubtleText, p_envStatusLabel)));
        } else {
            p_envStatusLabel->setText(tr("System Python not found"));
            p_envStatusLabel->setStyleSheet(
                QString("color: %1;").arg(ThemeColors::getCSSColor(
                    ThemeColors::StatusError, p_envStatusLabel)));
        }
    } else {
        // resolvePythonExecutable returns "python3" as fallback when nothing is found
        if (exe == QStringLiteral("python3") && !version.isEmpty()) {
            // The env path didn't contain an interpreter but system python3 works
            p_envStatusLabel->setText(tr("No interpreter found in environment; falling back to system Python"));
            p_envStatusLabel->setStyleSheet(
                QString("color: %1;").arg(ThemeColors::getCSSColor(
                    ThemeColors::StatusWarning, p_envStatusLabel)));
        } else if (!version.isEmpty()) {
            p_envStatusLabel->setText(version);
            p_envStatusLabel->setStyleSheet(
                QString("color: %1;").arg(ThemeColors::getCSSColor(
                    ThemeColors::StatusSuccess, p_envStatusLabel)));
        } else {
            p_envStatusLabel->setText(tr("No Python found"));
            p_envStatusLabel->setStyleSheet(
                QString("color: %1;").arg(ThemeColors::getCSSColor(
                    ThemeColors::StatusError, p_envStatusLabel)));
        }
    }
}

QString PythonSettingsWidget::getPythonVersion(const QString &exe)
{
    QProcess proc;
    proc.start(exe, {QStringLiteral("--version")});
    if (!proc.waitForFinished(3000))
        return {};
    // Python 3 writes to stdout; Python 2 writes to stderr
    QByteArray out = proc.readAllStandardOutput().trimmed();
    if (!out.isEmpty())
        return QString::fromUtf8(out);
    return QString::fromUtf8(proc.readAllStandardError().trimmed());
}
