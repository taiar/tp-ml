<?php

  class run_analys {

    private $execution_size;
    private $epochs;
    private $classes;
    private $extra;

    private $file;
    private $file_source;
    private $file_size;
    private $file_line;

    public $standard_deviation_epoch;
    public $mean_epoch;
    public $standard_deviation_class;

    private $num_runs = 0;

    public $runs;

    public function __construct($file = '', $classes = 10, $epochs = 30, $extra = 8) {
      $this->classes = $classes;
      $this->epochs = $epochs;
      $this->extra = $extra;
      $this->execution_size = $this->extra + $this->classes + $this->epochs;
      if(!empty($file)) $this->set_file($file);
    }

    public function set_file($file) {
      $this->file = $file;
      $this->file_source = file($this->file);
      $this->file_size = count($this->file_source);
      $this->file_line = 0;
      $this->num_runs = $this->file_size / $this->execution_size;
      $this->runs = array();
    }

    public function read() {
      for ($i=0; $i < $this->num_runs; $i++) {
        $run = new single_run();
        $this->line_rev(5);
        for ($j=0; $j < $this->epochs; $j++) {
          $run->read_epoch($this->line_read());
          $this->line_next();
        }
        $this->line_rev(3);
        for ($j=0; $j < $this->classes; $j++) {
          $run->read_class($this->line_read());
          $this->line_next();
        }
        $this->runs[] = $run;
      }
      $this->file_source = array();
    }

    public function proccess() {
      $this->standard_deviation_epoch = array();
      $arr = array();
      $mean = 0;
      for ($i=0; $i < $this->epochs; $i++) {
        $mean = 0;
        $arr = array();
        for ($j=0; $j < $this->num_runs; $j++) {
          $arr[] = $this->runs[$j]->get_epoch($i)['por'];
          $mean += $this->runs[$j]->get_epoch($i)['por'];
        }
        $mean = $mean / $this->num_runs;
        $this->mean_epoch[$i] = $mean;
        $this->standard_deviation_epoch[$i] = $this->deviation($arr, $mean);
      }
    }

    private function deviation($arr, $mean) {
      $n = count($arr);
      $variance = 0;
      for ($i=0; $i < $n; $i++)
        $variance += ($arr[$i] - $mean)^2;
      $variance = $variance / $this->num_runs;
      return sqrt($variance);
    }

    private function line_read() {
      return $this->file_source[$this->file_line];
    }

    private function line_next() {
      $this->file_line += 1;
    }

    private function line_rev($n) {
      $this->file_line += $n;
    }
  }

  class single_run {

    private $execution_size;
    private $epochs;
    private $classes;
    private $extra;

    private $epoch = array();
    private $class = array();

    public function __construct($classes = 10, $epochs = 30, $extra = 8) {
      $this->classes = $classes;
      $this->epochs = $epochs;
      $this->extra = $extra;
      $this->execution_size = $this->extra + $this->classes + $this->epochs;

      $this->epoch = array();
      $this->class = array();
    }

    public function read_epoch($s) {
      $base = strpos($s, "had ") + 4;
      $base_2 = strpos($s, "errors") + 8;
      $this->epoch[] = array(
        'num' => substr($s, $base, strpos(substr($s, $base), " ")),
        'por' => substr($s, $base_2, 6)
      );
    }

    public function read_class($s) {
      $base = strpos($s, "/");
      $this->class[] = array(
        'num' => substr($s, 3, $base - 3),
        'tot' => substr($s, $base + 1, strpos(substr($s, $base), "(") - 1),
        'por' => substr($s, strpos($s, "(") + 1, 6)
      );
    }

    public function get_epoch($i) {
      return $this->epoch[$i];
    }

    public function get_class($i) {
      return $this->class[$i];
    }
  }

  require_once ('jpgraph/src/jpgraph.php');
  require_once ('jpgraph/src/jpgraph_log.php');
  require_once ('jpgraph/src/jpgraph_bar.php');

  $a = new run_analys('3_4_saida_cnn');
  $a->read();
  $a->proccess();

  $b = new run_analys('3_4_saida_mlp');
  $b->read();
  $b->proccess();

  $graph_a = new Graph(800, 500, 'auto');
  $graph_a->SetScale("intint");
  $graph_a->SetY2Scale("log");

  $theme_class = new UniversalTheme;
  $graph_a->SetTheme($theme_class);

  $graph_a->yaxis->SetTickPositions(array(50,60,70,80,90,100),
    array(55,65,75,85,95));

  $graph_a->SetBox(false);

  $graph_a->ygrid->SetFill(false);
  $graph_a->xaxis->SetTickLabels(array('1','2','3','4','5','6','7','8','9','10','11','12','13','14',
    '15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'));
  $graph_a->yaxis->HideLine(false);
  $graph_a->yaxis->HideTicks(false,false);

  // Create the bar plots
  $b1plot = new BarPlot($a->mean_epoch);
  $b2plot = new BarPlot($b->mean_epoch);

  // Create the grouped bar plot
  $gbplot = new GroupBarPlot(array($b1plot,$b2plot));
  $graph_a->Add($gbplot);

  $b1plot->SetColor("white");
  $b1plot->SetFillColor("#cc1111");

  $b2plot->SetColor("white");
  $b2plot->SetFillColor("#11cccc");

  $graph_a->title->Set("CNNs");

  $graph_a->Stroke(dirname(__FILE__) . '/cnn.png');

  // echo print_r($a->mean_epoch);
  // echo print_r($a->standard_deviation_epoch);
  // echo print_r($b->mean_epoch);
  // echo print_r($b->standard_deviation_epoch);