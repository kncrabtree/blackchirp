#include "motordisplaywidget.h"
#include "ui_motordisplaywidget.h"

#include <algorithm>

MotorDisplayWidget::MotorDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MotorDisplayWidget)
{
    ui->setupUi(this);

    ui->zSlider1->setAxis(BlackChirp::MotorX);
    ui->zSlider2->setAxis(BlackChirp::MotorT);
    ui->xySlider1->setAxis(BlackChirp::MotorZ);
    ui->xySlider2->setAxis(BlackChirp::MotorT);
    ui->timeXSlider->setAxis(BlackChirp::MotorX);
    ui->timeYSlider->setAxis(BlackChirp::MotorY);
    ui->timeZSlider->setAxis(BlackChirp::MotorZ);

    d_sliders << ui->zSlider1 << ui->zSlider2 << ui->xySlider1 << ui->xySlider2
              << ui->timeXSlider << ui->timeYSlider << ui->timeZSlider;
}

MotorDisplayWidget::~MotorDisplayWidget()
{
    delete ui;
}

void MotorDisplayWidget::prepareForScan(const MotorScan s)
{
    d_currentScan = s;

    Q_FOREACH(MotorSliderWidget *w,d_sliders)
        w->setRange(s);

    ui->motorZSpectrogramPlot->prepareForScan(s);
    ui->motorTimePlot->prepareForScan(s);
    ui->motorXYSpectrogramPlot->prepareForScan(s);
}

void MotorDisplayWidget::newMotorData(const MotorScan s)
{
    d_currentScan = s;

    //prepare slices; update plots
    QVector<double> sliceZPlot = s.slice(ui->motorZSpectrogramPlot->leftAxis(),ui->motorZSpectrogramPlot->bottomAxis()
                                         ,ui->zSlider1->axis(),ui->zSlider1->currentIndex(),
                                         ui->zSlider2->axis(),ui->zSlider2->currentIndex());

    ui->motorZSpectrogramPlot->updateData(sliceZPlot,s.numPoints(ui->motorZSpectrogramPlot->bottomAxis()));

    QVector<double> sliceXYPlot = s.slice(ui->motorXYSpectrogramPlot->leftAxis(),ui->motorXYSpectrogramPlot->bottomAxis()
                                         ,ui->xySlider1->axis(),ui->xySlider1->currentIndex(),
                                         ui->xySlider2->axis(),ui->xySlider2->currentIndex());
    ui->motorXYSpectrogramPlot->updateData(sliceXYPlot,s.numPoints(ui->motorXYSpectrogramPlot->bottomAxis()));

    QVector<QPointF> timeTrace = s.tTrace(ui->timeXSlider->currentIndex(),ui->timeYSlider->currentIndex(),ui->timeZSlider->currentIndex());
    ui->motorTimePlot->updateData(timeTrace);

}
