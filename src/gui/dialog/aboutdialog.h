#ifndef ABOUTDIALOG_H
#define ABOUTDIALOG_H

#include <QDialog>
#include <QList>
#include <QPair>
#include <QString>

class AboutDialog : public QDialog
{
    Q_OBJECT
public:
    struct AppInfo {
        QString name;
        QString version;
        QString build;
        QString description;
        QList<QPair<QString,QString>> features;
    };

    explicit AboutDialog(const AppInfo &info, QWidget *parent = nullptr);

protected:
    void showEvent(QShowEvent *event) override;
};

#endif // ABOUTDIALOG_H
